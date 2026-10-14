"""원클릭 파이프라인 음질 검증 리포트 (실제 AI 모델 · 실제 키 변경 엔진 사용).

정답 트랙이 있는 MUSDB18 7초 샘플 4곡을 이어 붙여 '원곡'을 만들고, 이를 YouTube 오디오 스트림처럼
48kHz Opus 160kbps(WebM, itag 251 과 같은 형식)로 인코딩한 파일을 입력으로 씁니다. 그 뒤 앱과 똑같은 경로
(pipeline.Pipeline)로 디코딩 → 분리 → 키 변경 → 저장한 결과를 다시 읽어 확인합니다.

확인 항목
- 입력: 코덱·샘플레이트·채널, 리샘플링 횟수(AI 입력용 1회뿐인지), 중간 손실 인코딩 여부
- 분리: 정답 대비 SDR(보컬/MR), 스테레오 유지, 길이, 클리핑, 보컬 + MR = 원본(같은 샘플레이트에서)
- 키: 0 반음이면 피치 처리 미실행 + 결과 비트 동일, ±2/−3/±12 음정 이동량·길이·박자(타이밍) 유지
- 출력: WAV 24-bit / FLAC(= WAV 와 샘플 동일) / MP3 320kbps CBR, 모든 파일 다시 디코딩 가능

사용:
  MUSIC_SEPARATOR_MODELS_DIR=… python scripts/quality_report.py --data build/musdb7 --out quality_report.json
  (--preset speed 로 Demucs 보컬 모델 경로 확인)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 44100


def sdr(ref, est):
    ref = ref.astype(np.float64)
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-9) / (np.sum((ref - est) ** 2) + 1e-9)))


def width_db(x):
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((np.sum(side.astype(np.float64) ** 2) + 1e-9) / (np.sum(mid.astype(np.float64) ** 2) + 1e-9)))


def logspec(x, sr, fmin=110.0, fmax=3520.0, bins_per_st=8):
    """시간 평균 로그 주파수 스펙트럼 (반음당 8칸)."""
    mono = x.mean(1).astype(np.float64)
    n = 8192
    hop = 2048
    win = np.hanning(n)
    frames = [np.abs(np.fft.rfft(mono[i:i + n] * win)) for i in range(0, len(mono) - n, hop)]
    mag = np.mean(frames, 0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    nb = int(round(12 * bins_per_st * np.log2(fmax / fmin)))
    centers = fmin * 2 ** (np.arange(nb) / (12 * bins_per_st))
    spec = np.interp(centers, freqs, mag)
    return np.log(spec + 1e-6)


def whitened(x, sr, bins_per_st=8):
    """로그 스펙트럼에서 ±반옥타브 이동 평균(스펙트럼 기울기·포락선)을 빼서 배음 봉우리만 남김.
    드럼처럼 넓은 대역 소리가 많은 MR 에서도 음정 이동량을 잴 수 있게 함."""
    ls = logspec(x, sr, bins_per_st=bins_per_st)
    w = bins_per_st * 6
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return ls - np.convolve(np.pad(ls, w, mode="edge"), k, mode="valid")


def estimate_shift(ref, est, sr, bins_per_st=8, max_st=13):
    """두 신호의 음정 차이(반음). 배음 봉우리 위치를 로그 주파수 축에서 맞춰 봄."""
    a, b = whitened(ref, sr, bins_per_st), whitened(est, sr, bins_per_st)
    best, best_k = -1e9, 0
    for k in range(-max_st * bins_per_st, max_st * bins_per_st + 1):
        if k >= 0:
            c = np.dot(a[:len(a) - k], b[k:]) / max(1, len(a) - k)
        else:
            c = np.dot(a[-k:], b[:len(b) + k]) / max(1, len(b) + k)
        if c > best:
            best, best_k = c, k
    return best_k / bins_per_st


def onset_env(x, sr, hop=256):
    mono = x.mean(1).astype(np.float64)
    n = 1024
    win = np.hanning(n)
    prev = None
    env = []
    for i in range(0, len(mono) - n, hop):
        m = np.abs(np.fft.rfft(mono[i:i + n] * win))
        env.append(0.0 if prev is None else float(np.sum(np.maximum(0, m - prev))))
        prev = m
    return np.array(env)


def timing_lag_ms(ref, est, sr, hop=256, max_ms=60):
    a, b = onset_env(ref, sr, hop), onset_env(est, sr, hop)
    a, b = a - a.mean(), b - b.mean()
    m = int(max_ms / 1000 * sr / hop)
    lags = range(-m, m + 1)
    vals = [np.dot(a[max(0, -k):len(a) - max(0, k)], b[max(0, k):len(b) - max(0, -k)]) for k in lags]
    return float(list(lags)[int(np.argmax(vals))] * hop / sr * 1000)


def ffprobe(path):
    from music_separator import audio_io

    return audio_io.probe_audio(path)


def mp3_is_cbr(path):
    """모든 MPEG 프레임 헤더의 비트레이트가 320kbps 인지 (Xing/Info 첫 프레임 제외)."""
    data = Path(path).read_bytes()
    rates = {}
    i = 0
    br_table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    sr_table = [44100, 48000, 32000, 0]
    n = 0
    while i < len(data) - 4:
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            ver = (data[i + 1] >> 3) & 3
            layer = (data[i + 1] >> 1) & 3
            if ver == 3 and layer == 1:
                bri = data[i + 2] >> 4
                sri = (data[i + 2] >> 2) & 3
                pad = (data[i + 2] >> 1) & 1
                br, srr = br_table[bri], sr_table[sri]
                if br and srr:
                    size = 144000 * br // srr + pad
                    rates[br] = rates.get(br, 0) + 1
                    n += 1
                    i += size
                    continue
        i += 1
    return rates, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "build" / "musdb7"))
    ap.add_argument("--tracks", type=int, default=4)
    ap.add_argument("--preset", default="quality")
    ap.add_argument("--out", default="quality_report.json")
    a = ap.parse_args()

    from evaluate_separation import fetch, load_tracks

    from music_separator import audio_io, engine, pipeline, pitch, system

    engine.configure_threads()
    work = Path(tempfile.mkdtemp(prefix="mr-quality-"))
    tracks = load_tracks(fetch(Path(a.data)), a.tracks)
    voc = np.concatenate([t["vocals"] for t in tracks])
    inst = np.concatenate([t["inst"] for t in tracks])
    mix = voc + inst
    print(f"원곡: MUSDB18 7초 샘플 {len(tracks)}곡 = {len(mix) / SR:.1f}초", flush=True)

    # YouTube 와 같은 형식의 입력 만들기: 48kHz Opus 160kbps WebM
    mix48 = audio_io.resample(mix, SR, 48000)
    voc48 = audio_io.resample(voc, SR, 48000)
    inst48 = audio_io.resample(inst, SR, 48000)
    src = work / "youtube_like.webm"
    p = subprocess.run([audio_io.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", "48000",
                        "-ac", "2", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "160k", str(src)],
                       input=np.ascontiguousarray(mix48, "<f4").tobytes(), capture_output=True)
    if p.returncode:
        raise SystemExit(p.stderr.decode())
    rep: dict = {"source": "MUSDB18-7 test " + ", ".join(t["name"] for t in tracks), "checks": {}}
    checks = rep["checks"]

    # 계측: 키 변경 호출 횟수, MP3 인코딩 횟수
    calls = {"pitch": 0, "mp3": 0, "resample": []}
    orig_shift, orig_mp3, orig_rs = pitch.shift, audio_io.write_mp3, audio_io.resample

    def shift_count(*args, **kw):
        calls["pitch"] += 1
        return orig_shift(*args, **kw)

    def mp3_count(*args, **kw):
        calls["mp3"] += 1
        return orig_mp3(*args, **kw)

    def rs_log(data, s, d):
        if s != d:
            calls["resample"].append(f"{s}->{d}")
        return orig_rs(data, s, d)
    pitch.shift, audio_io.write_mp3, audio_io.resample = shift_count, mp3_count, rs_log

    def run(target, semis, fmt, prev=None):
        spec = pipeline.JobSpec(str(src), work / "out", target=target, semitones=semis, fmt=fmt, preset=a.preset,
                                device=system.choose_device("auto").key)
        before = dict(pitch=calls["pitch"], mp3=calls["mp3"], rs=len(calls["resample"]))
        t0 = time.monotonic()
        pl = pipeline.Pipeline(spec, work / "pipe")
        r = pl.run(*(prev or ()))
        d = dict(pitch=calls["pitch"] - before["pitch"], mp3=calls["mp3"] - before["mp3"],
                 resample=calls["resample"][before["rs"]:], sec=round(time.monotonic() - t0, 1))
        return r, d

    # 1) MR · WAV 24 · 키 0  (분리 1회)
    r0, d0 = run("mr", 0, "wav24")
    prev = (r0.prepared, r0.separation, pipeline.separation_key(r0.prepared.source_id, r0.separation.model,
                                                                r0.separation.preset))
    info = r0.prepared.info
    checks["input"] = {"codec": info.codec, "sample_rate": info.sample_rate, "channels": info.channels,
                       "bit_rate_kbps": info.bit_rate_kbps, "container": info.container,
                       "resample_steps": r0.prepared.resample_steps,
                       "pipeline_resample_calls": d0["resample"],
                       "resample_note": "48000->44100 = AI 입력(원본 1회), 44100->48000 = 보컬 추정치만 (MR = 원본 − 보컬 계산용, "
                                        "남는 악기는 리샘플링 없음)",
                       "lossy_encodes": d0["mp3"]}
    sep = r0.separation
    checks["separation"] = {"engine": sep.engine, "model": sep.model, "preset": sep.preset, "device": sep.device,
                            "elapsed_sec": round(sep.elapsed, 1),
                            "realtime_factor": round(sep.elapsed / (len(mix) / SR), 2)}
    mr_path = r0.files[0]
    mr = audio_io.decode_audio(mr_path, None)
    det = r0.export.details[0]
    vocals44 = np.load(sep.stems[0].path)
    n48 = len(mix48)
    checks["mr_wav24"] = {
        "file": mr_path.name, "probe": ffprobe(mr_path).summary(), "sample_rate": det.sample_rate,
        "length_ok": abs(len(mr) - n48) <= 1, "channels": mr.shape[1],
        "sdr_vs_truth_db": round(sdr(inst48[:len(mr)], mr[:len(inst48)]), 2),
        "width_err_db": round(abs(width_db(mr) - width_db(inst48)), 3),
        "peak_dbfs": round(det.peak_db, 2), "gain_db": round(r0.export.gain_db, 2),
        "pitch_calls": d0["pitch"]}
    # 보컬 + MR = 원본 (48kHz, 디코딩한 입력 기준)
    native = np.load(r0.prepared.native_path)
    up_voc = audio_io.resample(vocals44, SR, 48000)[:len(native)]
    g = 10 ** (r0.export.gain_db / 20)
    m = min(len(native), len(mr), len(up_voc))
    recon = mr[:m] / g + up_voc[:m]
    checks["mr_wav24"]["vocals_plus_mr_minus_original_dbfs"] = round(
        float(20 * np.log10(np.max(np.abs(recon - native[:m])) + 1e-12)), 1)

    # 2) 보컬만 · FLAC (분리 재사용) + WAV24 동일성
    r1, d1 = run("vocals", 0, "flac", prev)
    r1w, _ = run("vocals", 0, "wav24", prev)
    vf = audio_io.decode_audio(r1.files[0], None)
    vw = audio_io.decode_audio(r1w.files[0], None)
    voc_rate = r1.export.details[0].sample_rate
    truth_v = voc if voc_rate == SR else voc48
    checks["vocals_flac"] = {
        "file": r1.files[0].name, "probe": ffprobe(r1.files[0]).summary(), "sample_rate": voc_rate,
        "reused_separation": r1.reused_separation, "flac_equals_wav24": bool(np.array_equal(vf, vw)),
        "sdr_vs_truth_db": round(sdr(truth_v[:len(vf)], vf[:len(truth_v)]), 2),
        "length_ok": abs(len(vf) - len(truth_v)) <= 1, "channels": vf.shape[1], "lossy_encodes": d1["mp3"]}

    # 3) MR · MP3 320
    r2, d2 = run("mr", 0, "mp3", prev)
    rates, nframes = mp3_is_cbr(r2.files[0])
    m3 = audio_io.decode_audio(r2.files[0], None)
    checks["mr_mp3"] = {"file": r2.files[0].name, "probe": ffprobe(r2.files[0]).summary(),
                        "frame_bitrates": {str(k): v for k, v in rates.items()}, "frames": nframes,
                        "cbr_320": set(rates) <= {320} and nframes > 0, "mp3_encodes": d2["mp3"],
                        "true_peak_dbtp": r2.export.details[0].true_peak_db,
                        "decodable_length_ok": abs(len(m3) - n48) <= 48, "lossy_encodes_total": d2["mp3"]}

    # 4) 키 변경: 0 은 처리 생략·비트 동일, ±k 는 음정·길이·타이밍
    r3, d3 = run("mr", 0, "wav24", prev)
    same = np.array_equal(audio_io.decode_audio(r3.files[0], None), mr)
    checks["key_0"] = {"pitch_calls": d3["pitch"], "bit_identical_to_previous": bool(same)}
    keys = {}
    for k in (2, -3, 12, -12):
        rk, dk = run("mr", k, "wav24", prev)
        y = audio_io.decode_audio(rk.files[0], None)
        keys[f"{k:+d}"] = {"pitch_calls": dk["pitch"], "length_ok": abs(len(y) - len(mr)) <= 1,
                           "measured_shift_st": estimate_shift(mr, y, 48000),
                           "timing_lag_ms": timing_lag_ms(mr, y, 48000),
                           "peak_dbfs": round(rk.export.details[0].peak_db, 2), "gain_db": round(rk.export.gain_db, 2),
                           "engine": rk.export.key_engine}
    rv, dv = run("vocals", -3, "wav24", prev)
    yv = audio_io.decode_audio(rv.files[0], None)
    keys["vocals -3 (포먼트 유지)"] = {"pitch_calls": dv["pitch"], "length_ok": abs(len(yv) - len(vw)) <= 1,
                                   "measured_shift_st": estimate_shift(vw, yv, voc_rate),
                                   "timing_lag_ms": timing_lag_ms(vw, yv, voc_rate),
                                   "formant": rv.export.details[0].formant}
    checks["key"] = keys
    pitch.shift, audio_io.write_mp3, audio_io.resample = orig_shift, orig_mp3, orig_rs

    # 작업 폴더 안에 MP3 같은 중간 손실 파일이 없는지
    checks["intermediate_files"] = sorted({p.suffix for p in (work / "pipe").rglob("*") if p.is_file()})
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
