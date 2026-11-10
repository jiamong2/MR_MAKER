"""MR SEPARATOR / MR MAKER 화면 동작 확인 (offscreen, 가짜 분리 결과 — AI 모델 없이 실행).

사용: python tests/separator_ui_check.py <작업 폴더> [separator|maker] [--empty]
test_separator.py 가 에디션별로 따로 프로세스를 띄워 실행한다 (에디션은 프로세스 시작 때 정해지므로).
"""
import os, sys, time, json
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MUSIC_SEPARATOR_EDITION"] = sys.argv[2] if len(sys.argv) > 2 else "separator"
os.environ["MUSIC_SEPARATOR_NONINTERACTIVE"] = "1"
work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
os.environ["MUSIC_SEPARATOR_DATA_DIR"] = str(work / "data")
os.environ["MUSIC_SEPARATOR_CACHE_DIR"] = str(work / "cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PySide6.QtCore import QSettings, QCoreApplication
from PySide6.QtWidgets import QApplication
import music_separator as ms
from music_separator import audio_io, engine
from music_separator.dsp import compute_peaks
from music_separator.ui import theme
from music_separator.ui.main_window import MainWindow, MODE
from music_separator.ui.workers import LoadedAudio

QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(work / "ini"))
QCoreApplication.setOrganizationName(ms.ORG_NAME); QCoreApplication.setApplicationName(ms.SETTINGS_ID)
app = QApplication([]); theme.apply(app); theme.finalize_stylesheet(app)
w = MainWindow(QSettings()); w.resize(1500, 1000); w.show(); app.processEvents()
print("edition", ms.EDITION, ms.APP_NAME, "MODE", MODE, "page", w.pages.currentIndex(), "preset", w.preset_key())
w.output_dir = work / "out"

sr = 44100; n = sr * 20; t = np.arange(n) / sr
rng = np.random.default_rng(1)
def st(x): return np.stack([x, x], 1).astype(np.float32)
env = lambda f: (np.sin(2*np.pi*f*t) > 0).astype(np.float32)
stems = {
 "vocals": st(0.20*np.sin(2*np.pi*440*t) * (0.6+0.4*np.sin(2*np.pi*0.3*t))),
 "drums":  st(0.25*rng.standard_normal(n) * np.exp(-((t*2) % 1)*12)),
 "bass":   st(0.25*np.sin(2*np.pi*55*t) * env(0.5)),
 "guitar": st(0.15*np.sign(np.sin(2*np.pi*196*t)) * env(0.25)),
 "piano":  st(0.20*np.sin(2*np.pi*262*t) * np.exp(-((t*1.5) % 1)*4)),
 "other":  st(0.10*np.sin(2*np.pi*660*t) * env(0.125)),
}
mix = sum(stems.values()).astype(np.float32)
d = w.session_dir; np.save(d/"orig.npy", mix)
run = d/"run1"; run.mkdir()
src = work/"Blue Night (Live Session).wav"; audio_io.write_wav(src, mix, 24, sr)
la = LoadedAudio(src, d/"orig.npy", n, compute_peaks(mix), sr, d/"orig.npy", source_id="file:test")
w._on_loaded(la); app.processEvents()
if "--empty" in sys.argv:
    w.grab().save(str(work/"shot-empty.png")); sys.exit(0)
res_stems = []
for k in engine.MODES[MODE].outputs:
    a = stems[k] if k in stems else (mix - stems["vocals"])
    np.save(run/f"{k}.npy", a); res_stems.append(engine.StemResult(k, run/f"{k}.npy", compute_peaks(a)))
res = engine.SeparationResult(MODE, "bs_roformer_sw", "cpu", res_stems, n, n/sr, 83.0, [], "roformer", 0.0, "quality")
w._on_separated(res); app.processEvents()
def wait():
    t0 = time.time()
    while w.busy() and time.time() - t0 < 120:
        app.processEvents(); time.sleep(0.02)
    app.processEvents()
out = {}
if MODE == "6":
    assert list(w.chips) == list(engine.MODES["6"].outputs), list(w.chips)
    assert w.mix_save_btn.isEnabled() and "전체" in w.mix_state.text(), w.mix_state.text()
    # 1) 칩으로 피아노 빼기 ↔ 트랙 음소거 동기화
    w.chips["piano"].click(); app.processEvents()
    assert w.mixer.track("piano").mute and w.rows["piano"].card.mute_btn.isChecked()
    assert w._mix_plan()[1] == ["piano"], w._mix_plan()
    out["state1"] = w.mix_state.text()
    w.grab().save(str(work/"shot-piano-off.png"))
    w.set_format("wav24"); w.export_mix(); wait()
    # 2) 트랙 M 버튼 → 칩 반영, 솔로
    w.rows["piano"].set_muted(False); app.processEvents()
    assert not w.chips["piano"].isChecked()
    w.rows["piano"].set_solo(True); app.processEvents()
    assert w._mix_plan()[0] == ["piano"], w._mix_plan()
    out["state2"] = w.mix_state.text()
    w.grab().save(str(work/"shot-piano-solo.png"))
    w.export_mix(); wait()
    # 3) 드럼+베이스 솔로, 베이스 볼륨 70%
    w.rows["piano"].set_solo(False); w.rows["drums"].set_solo(True); w.rows["bass"].set_solo(True)
    w.rows["bass"].slider.setValue(70); app.processEvents()
    out["state3"] = w.mix_state.text(); out["plan3"] = w._mix_plan()
    w.export_mix(); wait()
    # 4) 모두 켜기
    w.reset_mix(); app.processEvents()
    assert w._mix_plan() == (list(engine.MODES["6"].outputs), [], {}), w._mix_plan()
    assert w.rows["bass"].slider.value() == 100 and w.mixer.track("original").mute
    out["state4"] = w.mix_state.text()
    # 5) 전부 끄면 저장 버튼 비활성
    for k in w.chips: w.rows[k].set_muted(True)
    app.processEvents(); assert not w.mix_save_btn.isEnabled(); out["state5"] = w.mix_state.text()
    w.reset_mix(); app.processEvents()
    w.export_all(); wait()
    w.grab().save(str(work/"shot-main.png"))
    files = sorted((work/"out").rglob("*.wav"))
    out["files"] = [f.name for f in files]
    # 수치 확인: 믹스 파일 = 해당 트랙 합
    import wave
    def load(f):
        with wave.open(str(f)) as wv:
            b = wv.readframes(wv.getnframes()); sw = wv.getsampwidth(); ch = wv.getnchannels()
        a = np.frombuffer(b, dtype=np.uint8).reshape(-1, sw)
        v = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int8).astype(np.int32) << 16))
        return (v / 8388608.0).reshape(-1, ch).astype(np.float32)
    def err(name, want):
        f = [x for x in files if name in x.name][0]; a = load(f); m = min(len(a), len(want))
        g = float((a[:m] * want[:m]).sum() / (want[:m] ** 2).sum())      # 클리핑 방지 배율 허용
        out["gain_" + name] = round(g, 4)
        return float(np.abs(a[:m] - g * want[:m]).max())
    out["err_no_piano"] = err("피아노 제외", mix - stems["piano"])
    out["err_piano_only"] = err("믹스 (피아노만)", stems["piano"])
    out["err_drum_bass"] = err("드럼+베이스", stems["drums"] + 0.7 * stems["bass"])
    assert max(out["err_no_piano"], out["err_piano_only"], out["err_drum_bass"]) < 1e-4, out
    assert not w.home_btn.isVisible() and w.pages.currentIndex() == 1
    assert "홈 화면" not in [a.text() for a in w._menu.actions()]
else:
    assert w.pages.currentIndex() == 0 and w.mix_save_btn is None
    w.show_advanced(); app.processEvents()
    assert w.home_btn.isVisible() and list(w.rows) == ["original", "vocals", "instrumental"]
    w.grab().save(str(work/"shot-maker.png"))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("OK")
w.close()
