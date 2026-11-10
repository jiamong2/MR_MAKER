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
    # 왼쪽 순서: 1 곡 불러오기 · 2 키 · 3 템포 · 4 AI 음원 분리 (화면 위에는 파일 열기/YouTube 없음)
    left = w.findChild(type(w.track_scroll), "LeftScroll").widget().layout()
    titles = []
    for i in range(left.count()):
        wd = left.itemAt(i).widget()
        if wd is not None and wd.isVisibleTo(w):
            from PySide6.QtWidgets import QLabel
            titles.append([l.text() for l in wd.findChildren(QLabel) if l.objectName() in ("NumBadge",) or
                           l.property("role") == "cardTitle"][:2])
    out["left"] = titles
    assert [t[0] for t in titles] == ["1", "2", "3", "4"], titles
    assert "곡 불러오기" in titles[0][1] and "키" in titles[1][1] and "템포" in titles[2][1] and "AI 음원 분리" in titles[3][1]
    assert w.open_btn.window() is w and w.open_btn.parentWidget() is not w.play_btn.parentWidget()
    assert w.mixer.track("original").mute and not w.chips
    w.grab().save(str(work/"shot-main.png"))
    # 피아노만 빼기 → 모두 켜기
    w.rows["piano"].set_muted(True); app.processEvents()
    assert w.mixer.track("piano").mute
    w.reset_mix(); app.processEvents()
    assert not any(t.mute for t in w.mixer.tracks if t.key != "original") and w.mixer.track("original").mute
    w.set_format("wav24"); w.export_all(); wait()
    files = sorted((work/"out").rglob("*.wav"))
    out["files"] = [f.name for f in files]
    assert len(files) == 6, out["files"]
    assert "Blue Night" in w.source_lbl.text()
    assert not w.home_btn.isVisible() and w.pages.currentIndex() == 1
    assert "홈 화면" not in [a.text() for a in w._menu.actions()]
else:
    assert w.pages.currentIndex() == 0 and w.open_btn.parentWidget() is w.play_btn.parentWidget()
    w.show_advanced(); app.processEvents()
    assert w.home_btn.isVisible() and list(w.rows) == ["original", "vocals", "instrumental"]
    w.grab().save(str(work/"shot-maker.png"))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("OK")
w.close()
