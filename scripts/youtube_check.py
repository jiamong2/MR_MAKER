"""YouTube 오디오 획득 검증 (실제 네트워크 사용, CI 에서 실행).

크리에이티브 커먼즈(CC-BY) 영상으로
1. 선택된 오디오 스트림의 코덱·비트레이트·샘플레이트·채널 (yt-dlp 정보)
2. 받은 파일을 FFmpeg 로 다시 확인 (재인코딩 여부: 코덱·샘플레이트가 선택한 스트림과 같은지)
3. 원클릭 파이프라인으로 디코딩 → (초소형 테스트 모델) 분리 → 저장까지 실행해
   리샘플링 단계가 'AI 입력용 1회'뿐인지, 중간 손실 인코딩이 없는지 확인
을 출력합니다. YouTube 가 데이터센터 IP 를 차단하는 경우가 있어 실패해도 빌드를 막지 않습니다.

사용: python scripts/youtube_check.py [URL ...] [--annotate]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=aqz-KE-bpKQ",   # Big Buck Bunny (Blender Foundation, CC-BY 3.0)
    "https://www.youtube.com/watch?v=eRsGyueVLvQ",   # Sintel (Blender Foundation, CC-BY 3.0)
]


def main() -> int:
    from music_separator import audio_io, pipeline, youtube

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    annotate = "--annotate" in sys.argv
    urls = args or DEFAULT_URLS
    print(f"yt-dlp {youtube.ytdlp_version()} · deno {youtube.find_deno()}")
    ok_any = False
    report = []
    for url in urls:
        work = Path(tempfile.mkdtemp(prefix="ytcheck-"))
        row = {"url": url}
        try:
            a = youtube.acquire(url, work / "dl", lambda f, m: None)
            info = audio_io.probe_audio(a.path)
            row.update(format_id=a.format_id, codec=a.codec, abr_kbps=a.abr_kbps, asr=a.sample_rate,
                       channels=a.channels, audio_only=a.audio_only, container=a.container, reason=a.reason,
                       license=a.license, file=a.path.name, filesize=a.filesize,
                       probe=info.summary(), probe_codec=info.codec, probe_rate=info.sample_rate)
            same_codec = a.codec.split(".")[0].lower().startswith(info.codec.split()[0].lower()[:3]) or \
                (a.codec.startswith("mp4a") and info.codec.startswith("aac"))
            row["not_reencoded"] = bool(same_codec and info.sample_rate == a.sample_rate)
            row["candidates"] = a.table
            # 파이프라인 (초소형 모델로 디코딩/리샘플/저장 경로만 확인, 앞 40초)
            spec = pipeline.JobSpec(str(a.path), work / "out", target="mr", fmt="wav24",
                                    model_override="demucs_unittest")
            res = pipeline.Pipeline(spec, work / "pipe").run()
            row["pipeline_resample_steps"] = res.report["decode"]["resample_steps"]
            row["pipeline_output"] = res.report["outputs"][0]["verified"]
            row["lossy_encodes"] = res.report["lossy_encodes"]
            ok_any = True
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        report.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=1, default=str), flush=True)
        if annotate:
            if "error" in row:
                print(f"::warning title=youtube {url}::{row['error'][:500]}")
            else:
                print(f"::notice title=youtube {url}::stream {row['format_id']} {row['codec']} "
                      f"{row['abr_kbps']:g}kbps {row['asr']}Hz ch{row['channels']} audio_only={row['audio_only']} "
                      f"| file {row['probe']} | not_reencoded={row['not_reencoded']} | "
                      f"resample={row['pipeline_resample_steps']} | lossy_encodes={row['lossy_encodes']} | "
                      f"{row['reason']}")
    Path("youtube_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
