"""메인 윈도우."""
from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar,
                               QScrollArea, QScrollBar, QSizePolicy, QSlider, QStackedWidget,
                               QVBoxLayout, QWidget)

from .. import APP_BRAND, APP_NAME, APP_SHORT, __version__, audio_io, engine, models, paths, pitch, system
from ..dsp import audible_flags
from ..player import Mixer, Player, Track
from . import icons, theme
from .widgets import (COLUMN_GAP, HEADER_WIDTH, ROW_SPACING, ChipButton, DropZone, Lcd, LogoTile, ModeTile,
                      PlayButton, TimeRuler, ToggleSwitch, TrackRow, TracksPanel, Viewport, glow, icon_button,
                      label, repolish, text_button)
from ..export import ExportJob, ExportOptions, ExportSource
from .workers import ExportThread, KeyThread, LoadedAudio, LoadThread, SeparationThread

log = logging.getLogger(__name__)

# (엔진, 모드) → (제목, 설명)
MODE_INFO = {
    ("demucs", "2"): ("보컬과 반주를 깔끔하게",
                      "노래만 남기거나 빼고 싶을 때 가장 깔끔한 모드입니다. 반주는 '원본 − 보컬'로 만들어 "
                      "악기 소리가 원본 그대로 남습니다."),
    ("demucs", "4"): ("가장 안정적인 기본 분리",
                      "Demucs가 가장 잘하는 분리입니다. 기타·피아노·신스 등은 '기타 악기' 트랙에 함께 들어갑니다."),
    ("demucs", "6"): ("악기를 더 세밀하게",
                      "보컬·드럼·베이스를 먼저 분리한 뒤 남은 소리에서 기타·피아노를 나눕니다. "
                      "피아노가 약하게 나오면 위에서 '고급 (BS-RoFormer)' 엔진을 선택해 보세요."),
    ("roformer", "2"): ("최신 모델로 보컬 분리",
                        "BS-RoFormer로 보컬을 분리합니다. 반주는 '원본 − 보컬'로 만들어 악기 소리가 원본 그대로 남습니다."),
    ("roformer", "4"): ("최신 모델로 4-Stem 분리",
                        "BS-RoFormer로 보컬·드럼·베이스를 분리합니다. 나머지 악기는 '기타 악기'에 모입니다."),
    ("roformer", "6"): ("피아노·기타까지 깔끔하게",
                        "BS-RoFormer가 6개 악기를 한 번에 분리합니다. 피아노 분리 품질이 Demucs보다 크게 좋습니다."),
}


class DeviceThread(QThread):
    done = Signal(list)

    def run(self):
        try:
            self.done.emit(system.detect_devices())
        except Exception as exc:
            log.exception("device detection failed")
            self.done.emit([system.DeviceInfo("cpu", f"CPU (장치 감지 실패: {exc})")])


class ModeSelector(QWidget):
    """2/4/6-Stem 타일 선택기."""

    changed = Signal(str)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, ModeTile] = {}
        for key, title, desc in (("2", "2-Stem", "보컬 / 반주"), ("4", "4-Stem", "보컬 / 드럼 /\n베이스 / 기타"),
                                 ("6", "6-Stem", "보컬 / 드럼 /\n베이스 / 기타 /\n피아노 / 나머지")):
            t = ModeTile(title, desc)
            self.group.addButton(t)
            self.buttons[key] = t
            t.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            lay.addWidget(t)

    def set_current(self, key: str):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def current(self) -> str:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return "4"


def card(number: str, title: str, caps: str) -> tuple[QFrame, QVBoxLayout]:
    c = QFrame()
    c.setObjectName("Card")
    lay = QVBoxLayout(c)
    lay.setContentsMargins(18, 18, 18, 18)
    lay.setSpacing(14)
    head = QHBoxLayout()
    head.setSpacing(12)
    n = QLabel(number)
    n.setObjectName("NumBadge")
    n.setFixedSize(34, 34)
    tcol = QVBoxLayout()
    tcol.setSpacing(1)
    tcol.addWidget(label(title, "cardTitle"))
    tcol.addWidget(label(caps, "caps"))
    head.addWidget(n, 0, Qt.AlignmentFlag.AlignTop)
    head.addLayout(tcol, 1)
    lay.addLayout(head)
    return c, lay


class SettingsDialog(QDialog):
    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("설정")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)
        lay.addWidget(label("저장 폴더", "sectionLabel"))
        row = QHBoxLayout()
        self.path_lbl = label(str(win.output_dir), "dim", wrap=True)
        self.path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        change = text_button("변경…", "folder", None, 36)
        change.clicked.connect(self._change)
        row.addWidget(self.path_lbl, 1)
        row.addWidget(change)
        lay.addLayout(row)
        lay.addWidget(label("곡마다 하위 폴더가 만들어지고, 그 안에 트랙 파일이 저장됩니다.", "faint", wrap=True))
        lay.addSpacing(6)
        lay.addWidget(label("음질", "sectionLabel"))
        self.native_box = QCheckBox("원본 샘플레이트로 저장 (권장)")
        self.native_box.setChecked(win.keep_native_rate)
        self.native_box.toggled.connect(win.set_keep_native_rate)
        lay.addWidget(self.native_box)
        lay.addWidget(label("AI는 44.1kHz에서 처리하지만, 원본이 48kHz·96kHz면 원래 샘플레이트로 되돌려 저장합니다. "
                            "MR의 남는 악기는 원본 그대로이고, 0dBFS를 넘는 파일은 자동으로 음량을 살짝 낮춰 "
                            "잘림(클리핑)을 막습니다.", "faint", wrap=True))
        lay.addSpacing(6)
        lay.addWidget(label("AI 모델", "sectionLabel"))
        mrow = QHBoxLayout()
        mrow.addWidget(label(f"{paths.user_models_dir()}", "dim", wrap=True), 1)
        b = text_button("폴더 열기", None, None, 36)
        b.clicked.connect(lambda: paths.open_in_file_manager(paths.user_models_dir()))
        mrow.addWidget(b)
        b2 = text_button("가져오기…", None, None, 36)
        b2.clicked.connect(win.import_models)
        mrow.addWidget(b2)
        lay.addLayout(mrow)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _change(self):
        self.win.choose_output_dir()
        self.path_lbl.setText(str(self.win.output_dir))


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[QSettings] = None):
        super().__init__()
        self.settings = settings or QSettings()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1220, 760)
        self._initial_size()
        self.setAcceptDrops(True)

        self.session_dir = paths.new_session_dir()
        self.run_counter = 0
        self.mixer = Mixer()
        self.player = Player(self.mixer)
        self.viewport = Viewport()
        self.loaded: Optional[LoadedAudio] = None
        self.result: Optional[engine.SeparationResult] = None
        self.rows: dict[str, TrackRow] = {}
        self.chips: dict[str, ChipButton] = {}
        self.devices: list[system.DeviceInfo] = []
        self.load_thread: Optional[LoadThread] = None
        self.sep_thread: Optional[SeparationThread] = None
        self.export_thread: Optional[ExportThread] = None
        self.key_thread: Optional[KeyThread] = None
        self.key_semitones = 0                 # 재생에 적용된 키
        self.key_target = 0                    # 사용자가 고른 키
        self.key_cache: dict = {}              # semitones -> {track: npy 경로}
        self.track_paths: dict = {}            # 원래(0키) 44.1kHz 트랙 경로
        self.keep_native_rate = True
        self.last_error: Optional[str] = None
        self._seeking = False

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._set_state()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play,
                  context=Qt.ShortcutContext.ApplicationShortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=self.rewind)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.rewind)
        QShortcut(QKeySequence.StandardKey.ZoomIn, self, activated=lambda: self.viewport.set_zoom(
            self.viewport.zoom * 1.5))
        QShortcut(QKeySequence.StandardKey.ZoomOut, self, activated=lambda: self.viewport.set_zoom(
            self.viewport.zoom / 1.5))

        self.device_thread = DeviceThread(self)
        self.device_thread.done.connect(self._on_devices)
        self.device_thread.start()

    def _initial_size(self):
        scr = QGuiApplication.primaryScreen()
        if scr is None:
            self.resize(1440, 900)
            return
        g = scr.availableGeometry()
        self.resize(min(1536, int(g.width() * 0.94)), min(1000, int(g.height() * 0.92)))

    # ================================================================ UI ===
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        body.addWidget(self._build_left_panel())
        right = QVBoxLayout()
        right.setSpacing(12)
        self.stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.drop_zone.browseRequested.connect(self.browse_file)
        self.stack.addWidget(self.drop_zone)
        self.stack.addWidget(self._build_tracks_panel())
        right.addWidget(self.stack, 1)
        right.addWidget(self._build_bottom_bar())
        body.addLayout(right, 1)
        rl.addLayout(body, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Header")
        bar.setFixedHeight(92)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(10)
        lay.addWidget(LogoTile())
        tcol = QVBoxLayout()
        tcol.setSpacing(3)
        trow = QHBoxLayout()
        trow.setSpacing(8)
        title = QLabel(APP_SHORT)
        title.setObjectName("AppTitle")
        brand = QLabel(APP_BRAND)
        brand.setObjectName("AppBrand")
        trow.addWidget(title, 0, Qt.AlignmentFlag.AlignBottom)
        trow.addWidget(brand, 0, Qt.AlignmentFlag.AlignBottom)
        trow.addStretch(1)
        self.file_title = QLabel("음악을 더 자유롭게, 당신의 사운드로")
        self.file_title.setObjectName("Tagline")
        self.file_title.setMaximumWidth(330)
        self.file_title.setMinimumWidth(10)
        tcol.addLayout(trow)
        tcol.addWidget(self.file_title)
        tw = QWidget()
        tw.setLayout(tcol)
        tw.setMinimumWidth(215)
        tw.setMaximumWidth(330)
        tw.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        lay.addWidget(tw)
        lay.addSpacing(10)

        self.rew_btn = icon_button("rewind", "처음으로 (Home)", 46, 18)
        self.rew_btn.clicked.connect(self.rewind)
        self.play_btn = PlayButton()
        self.play_btn.setToolTip("재생 / 일시정지 (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn = icon_button("stop", "정지", 46, 16)
        self.stop_btn.clicked.connect(self.stop)
        lay.addWidget(self.rew_btn)
        lay.addWidget(self.play_btn)
        lay.addWidget(self.stop_btn)
        lay.addSpacing(8)
        self.lcd = Lcd()
        lay.addWidget(self.lcd)
        lay.addSpacing(10)

        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setObjectName("Seek")
        self.seek.setRange(0, 10000)
        self.seek.setMinimumWidth(100)
        self.seek.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seek.setToolTip("재생 위치")
        self.seek.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.seek.sliderReleased.connect(self._seek_released)
        self.seek.valueChanged.connect(self._seek_moved)
        lay.addWidget(self.seek, 1)
        lay.addSpacing(14)

        vol = QLabel()
        vol.setPixmap(icons.pixmap("volume", theme.TEXT_DIM, 20))
        self.vol_icon = vol
        lay.addWidget(vol)
        self.master = QSlider(Qt.Orientation.Horizontal)
        self.master.setRange(0, 100)
        self.master.setValue(80)
        self.master.setFixedWidth(110)
        self.master.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.master.setToolTip("마스터 볼륨 (재생에만 적용)")
        self.master.valueChanged.connect(lambda v: setattr(self.mixer, "master", v / 100.0))
        self.mixer.master = 0.8
        lay.addWidget(self.master)
        lay.addSpacing(14)
        self.open_btn = text_button("파일 열기", "folder-fill", None, 46, theme.TEXT_DIM)
        self.open_btn.setMinimumWidth(124)
        self.open_btn.clicked.connect(self.browse_file)
        lay.addWidget(self.open_btn)
        self.more_btn = icon_button("dots", "메뉴", 46, 20)
        lay.addWidget(self.more_btn)
        return bar

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("LeftScroll")
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(384)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        panel.setObjectName("LeftPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 12, 0)
        lay.setSpacing(16)

        # ---- 1. 분리 ----
        c, cl = card("1", "AI 음원 분리", "AI MUSIC SEPARATION")
        self.mode_ctl = ModeSelector()
        self.mode_ctl.changed.connect(self._on_mode)
        cl.addWidget(self.mode_ctl)

        info = QFrame()
        info.setObjectName("InfoBox")
        il = QVBoxLayout(info)
        il.setContentsMargins(14, 12, 14, 12)
        il.setSpacing(6)
        ih = QHBoxLayout()
        ih.setSpacing(8)
        spark = QLabel()
        spark.setPixmap(icons.pixmap("sparkles", "#f5b400", 18))
        self.info_title = label("", "sectionLabel", wrap=True)
        ih.addWidget(spark)
        ih.addWidget(self.info_title, 1)
        self.info_text = label("", "dim", wrap=True)
        il.addLayout(ih)
        il.addWidget(self.info_text)
        cl.addWidget(info)

        cl.addWidget(label("AI 엔진", "sectionLabel"))
        self.engine_combo = QComboBox()
        for key, eng in engine.ENGINES.items():
            self.engine_combo.addItem(icons.icon("sparkles" if key == "roformer" else "layers", theme.TEXT_DIM, 18),
                                      eng.title, key)
        self.engine_combo.setIconSize(QSize(18, 18))
        self.engine_combo.currentIndexChanged.connect(lambda *_: self._on_mode(self.mode_ctl.current()))
        cl.addWidget(self.engine_combo)
        self.engine_hint = label("", "dim", wrap=True)
        cl.addWidget(self.engine_hint)

        hq = QHBoxLayout()
        hq.setSpacing(12)
        self.hq_box = ToggleSwitch()
        self.hq_box.toggled.connect(self._on_mode_options)
        hcol = QVBoxLayout()
        hcol.setSpacing(2)
        hcol.addWidget(label("고품질 모드 (처리 시간 증가)", "sectionLabel"))
        self.hq_hint = label("", "dim", wrap=True)
        hcol.addWidget(self.hq_hint)
        hq.addWidget(self.hq_box, 0, Qt.AlignmentFlag.AlignTop)
        hq.addLayout(hcol, 1)
        cl.addLayout(hq)

        cl.addWidget(label("처리 장치", "sectionLabel"))
        self.device_combo = QComboBox()
        cpu_icon = icons.icon("cpu", theme.TEXT_DIM, 18)
        self.device_combo.addItem(cpu_icon, "자동 선택 (권장)", "auto")
        self.device_combo.addItem(cpu_icon, "GPU", "gpu")
        self.device_combo.addItem(cpu_icon, "CPU", "cpu")
        self.device_combo.setIconSize(QSize(18, 18))
        self.device_combo.currentIndexChanged.connect(self._on_device_pref)
        cl.addWidget(self.device_combo)
        self.device_hint = label("처리 장치를 확인하는 중…", "dim", wrap=True)
        cl.addWidget(self.device_hint)

        self.start_btn = text_button("AI 분리 시작", "sparkles", "primary", 48)
        self.start_btn.clicked.connect(self.start_separation)
        glow(self.start_btn, theme.ACCENT, 20, 70, 3)
        cl.addWidget(self.start_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        cl.addWidget(self.progress)
        prow = QHBoxLayout()
        self.progress_lbl = label("", "dim", wrap=True)
        self.progress_lbl.hide()
        self.cancel_btn = text_button("취소", "x", "danger", 34, theme.DANGER)
        self.cancel_btn.clicked.connect(self.cancel_work)
        self.cancel_btn.hide()
        prow.addWidget(self.progress_lbl, 1)
        prow.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignTop)
        cl.addLayout(prow)
        self.notice_lbl = QLabel("")
        self.notice_lbl.setWordWrap(True)
        self.notice_lbl.setStyleSheet("background:#fff7ec; border:1px solid #f5d2a6; border-radius:10px;"
                                      " padding:8px 10px; color:#9a5a12; font-size:12px;")
        self.notice_lbl.hide()
        cl.addWidget(self.notice_lbl)
        lay.addWidget(c)

        # ---- 2. MR ----
        c, cl = card("2", "MR(반주) 만들기", "MR MAKER")
        cl.addWidget(label("빼고 싶은 악기를 누르세요. 원본에서 그 악기만 빼서 반주를 만들기 때문에 "
                           "남는 악기는 원본 음질 그대로입니다.", "dim", wrap=True))
        self.chip_box = QWidget()
        self.chip_grid = QGridLayout(self.chip_box)
        self.chip_grid.setContentsMargins(0, 0, 0, 0)
        self.chip_grid.setHorizontalSpacing(8)
        self.chip_grid.setVerticalSpacing(8)
        cl.addWidget(self.chip_box)
        self.chip_empty = label("분리를 마치면 여기에 악기 버튼이 나타납니다.", "faint", wrap=True)
        cl.addWidget(self.chip_empty)
        self.mr_volume_box = QCheckBox("트랙 볼륨 설정 반영")
        self.mr_volume_box.setToolTip("체크하면 각 트랙의 볼륨 슬라이더 값이 MR에 적용됩니다.")
        cl.addWidget(self.mr_volume_box)
        mr = QHBoxLayout()
        mr.setSpacing(10)
        self.mr_preview_btn = text_button("미리 듣기", "play", None, 46)
        self.mr_preview_btn.clicked.connect(self.preview_mr)
        self.mr_save_btn = text_button("MR 저장", "music", "primary", 46)
        self.mr_save_btn.clicked.connect(self.export_mr)
        glow(self.mr_save_btn, theme.ACCENT, 20, 70, 3)
        mr.addWidget(self.mr_preview_btn, 2)
        mr.addWidget(self.mr_save_btn, 3)
        cl.addLayout(mr)
        lay.addWidget(c)

        # ---- 3. 키 조절 ----
        c, cl = card("3", "키 조절", "KEY")
        krow = QHBoxLayout()
        krow.setSpacing(10)
        self.key_down = text_button("−", None, "step", 44)
        self.key_down.setFixedWidth(44)
        self.key_down.setToolTip("반음 내리기")
        self.key_down.clicked.connect(lambda: self.key_slider.setValue(self.key_slider.value() - 1))
        self.key_value = QLabel("0")
        self.key_value.setObjectName("KeyValue")
        self.key_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.key_value.setMinimumWidth(70)
        self.key_up = text_button("+", None, "step", 44)
        self.key_up.setFixedWidth(44)
        self.key_up.setToolTip("반음 올리기")
        self.key_up.clicked.connect(lambda: self.key_slider.setValue(self.key_slider.value() + 1))
        self.key_reset = text_button("원래 키", None, None, 36)
        self.key_reset.clicked.connect(lambda: self.key_slider.setValue(0))
        krow.addWidget(self.key_down)
        krow.addWidget(self.key_value, 1)
        krow.addWidget(self.key_up)
        krow.addSpacing(4)
        krow.addWidget(self.key_reset)
        cl.addLayout(krow)
        self.key_slider = QSlider(Qt.Orientation.Horizontal)
        self.key_slider.setRange(pitch.KEY_MIN, pitch.KEY_MAX)
        self.key_slider.setValue(0)
        self.key_slider.setPageStep(1)
        self.key_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.key_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.key_slider.setToolTip("반음 단위로 키를 조절합니다 (−12 ~ +12)")
        self.key_slider.valueChanged.connect(self._on_key_slider)
        cl.addWidget(self.key_slider)
        ends = QHBoxLayout()
        ends.addWidget(label("−12 (한 옥타브 ↓)", "faint"))
        ends.addStretch(1)
        ends.addWidget(label("+12 (한 옥타브 ↑)", "faint"))
        cl.addLayout(ends)
        self.keep_drums_box = QCheckBox("드럼은 원래 음높이 유지")
        self.keep_drums_box.setToolTip("드럼처럼 음정이 없는 소리는 그대로 두면 더 자연스럽습니다.\n"
                                       "(분리 후에 적용됩니다)")
        self.keep_drums_box.toggled.connect(self._on_keep_drums)
        cl.addWidget(self.keep_drums_box)
        self.key_hint = label("템포는 그대로, 음높이만 바뀝니다. 재생과 저장(MR·트랙)에 모두 적용됩니다.", "faint",
                              wrap=True)
        cl.addWidget(self.key_hint)
        self.key_timer = QTimer(self)
        self.key_timer.setSingleShot(True)
        self.key_timer.setInterval(450)       # 슬라이더를 움직이는 동안은 기다렸다가 처리
        self.key_timer.timeout.connect(self._apply_key)
        lay.addWidget(c)
        lay.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_tracks_panel(self) -> QWidget:
        panel = TracksPanel()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(10, 8, 10, 8)
        pl.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(COLUMN_GAP + 20)
        th = QLabel("트랙")
        th.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:14px; font-weight:600; padding:6px 0 0 8px;")
        th.setFixedWidth(HEADER_WIDTH)
        head.addWidget(th, 0, Qt.AlignmentFlag.AlignTop)
        tl = QVBoxLayout()
        tl.setSpacing(0)
        zrow = QHBoxLayout()
        zrow.setSpacing(4)
        zrow.addStretch(1)
        self.zoom_fit_btn = icon_button("zoom", "전체 보기", 30, 16, "ghost", theme.TEXT_DIM)
        self.zoom_fit_btn.clicked.connect(self.viewport.reset)
        zout = icon_button("minus", "축소 (Ctrl -)", 30, 16, "ghost", theme.TEXT_DIM)
        zout.clicked.connect(lambda: self.viewport.set_zoom(self.viewport.zoom / 1.5))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 100)
        self.zoom_slider.setFixedWidth(110)
        self.zoom_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.zoom_slider.setToolTip("확대 (Ctrl+휠로도 조절)")
        self.zoom_slider.valueChanged.connect(self._zoom_slider_moved)
        zin = icon_button("plus", "확대 (Ctrl +)", 30, 16, "ghost", theme.TEXT_DIM)
        zin.clicked.connect(lambda: self.viewport.set_zoom(self.viewport.zoom * 1.5))
        full = icon_button("expand", "전체 화면", 30, 16, "ghost", theme.TEXT_DIM)
        full.clicked.connect(self.toggle_fullscreen)
        for w in (self.zoom_fit_btn, zout, self.zoom_slider, zin, full):
            zrow.addWidget(w)
        tl.addLayout(zrow)
        self.ruler = TimeRuler(self.viewport)
        self.ruler.seekRequested.connect(self.seek_fraction)
        tl.addWidget(self.ruler)
        head.addLayout(tl, 1)
        pl.addLayout(head)

        self.track_scroll = QScrollArea()
        self.track_scroll.setWidgetResizable(True)
        self.track_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.track_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.track_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.track_scroll.viewport().setAutoFillBackground(False)
        self.track_container = QWidget()
        self.track_container.setAutoFillBackground(False)
        # 주의: 선택자 없는 스타일시트는 자식(M/S 버튼 등)에 상속되므로 objectName 으로 범위 지정
        self.track_container.setObjectName("TrackContainer")
        self.track_container.setStyleSheet("QWidget#TrackContainer { background: transparent; }")
        self.track_layout = QVBoxLayout(self.track_container)
        self.track_layout.setContentsMargins(0, 4, 0, 4)
        self.track_layout.setSpacing(ROW_SPACING)
        self.empty_hint = QLabel("왼쪽의 [AI 분리 시작]을 누르면 분리된 트랙이 여기에 나타납니다.")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setStyleSheet(f"color:{theme.TEXT_FAINT}; padding:30px; font-size:14px;")
        self.track_layout.addWidget(self.empty_hint)
        self.track_layout.addStretch(1)
        self.track_scroll.setWidget(self.track_container)
        pl.addWidget(self.track_scroll, 1)

        hrow = QHBoxLayout()
        hrow.addSpacing(HEADER_WIDTH + COLUMN_GAP + 20)
        self.hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self.hscroll.setRange(0, 0)
        self.hscroll.valueChanged.connect(lambda v: self.viewport.scroll_to(v / 10000))
        hrow.addWidget(self.hscroll, 1)
        pl.addLayout(hrow)
        self.viewport.changed.connect(self._on_viewport)
        self._on_viewport()
        return panel

    def _build_bottom_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("BottomBar")
        bar.setFixedHeight(70)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 12, 14, 12)
        lay.setSpacing(10)
        self.status_icon = QLabel()
        self.status_icon.setPixmap(icons.pixmap("check-circle", theme.SUCCESS, 22))
        self.status_state = QLabel("준비 완료")
        self.status_state.setObjectName("StatusOk")
        sep = QLabel("|")
        sep.setStyleSheet(f"color:{theme.BORDER_2};")
        self.status_msg = label("음악 파일을 열어 시작하세요.", "dim")
        self.status_msg.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.status_icon)
        lay.addWidget(self.status_state)
        lay.addWidget(sep)
        lay.addWidget(self.status_msg, 1)

        self.settings_btn = text_button("설정…", "settings", None, 44, theme.TEXT_DIM)
        self.settings_btn.clicked.connect(lambda: SettingsDialog(self).exec())
        self.folder_btn = icon_button("folder-fill", "저장 폴더 열기", 44, 18, "square", theme.TEXT_DIM)
        self.folder_btn.clicked.connect(self.open_output_dir)
        self.fmt_combo = QComboBox()
        for key, (lbl, _) in audio_io.EXPORT_FORMATS.items():
            self.fmt_combo.addItem(lbl, key)
        self.fmt_combo.setMinimumWidth(150)
        self.fmt_combo.setToolTip("저장 파일 형식")
        self.export_sel_btn = text_button("선택한 트랙 저장", "save-multi", None, 44)
        self.export_sel_btn.setToolTip("트랙 카드의 스위치가 켜진 트랙을 각각 파일로 저장합니다.\n"
                                       "개별 트랙은 볼륨 슬라이더와 관계없이 원래 음량으로 저장됩니다.")
        self.export_sel_btn.clicked.connect(self.export_selected)
        self.export_all_btn = text_button("전체 트랙 저장", "download", "primary", 44)
        self.export_all_btn.clicked.connect(self.export_all)
        glow(self.export_all_btn, theme.ACCENT, 22, 100, 3)
        for w in (self.settings_btn, self.folder_btn, self.fmt_combo, self.export_sel_btn, self.export_all_btn):
            lay.addWidget(w)
        return bar

    def _build_menu(self):
        menu = QMenu(self)
        acts = [
            ("파일 열기…", self.browse_file, QKeySequence.StandardKey.Open),
            ("저장 폴더 변경…", self.choose_output_dir, None),
            None,
            ("모든 음소거·솔로 해제", self.reset_mixer, None),
            ("전체 화면", self.toggle_fullscreen, QKeySequence.StandardKey.FullScreen),
            None,
            ("사용 방법", self.show_help, QKeySequence.StandardKey.HelpContents),
            ("AI 모델 폴더 열기", lambda: paths.open_in_file_manager(paths.user_models_dir()), None),
            ("AI 모델 가져오기 (오프라인 설치)…", self.import_models, None),
            ("로그 폴더 열기", lambda: paths.open_in_file_manager(paths.logs_dir()), None),
            None,
            (f"{APP_NAME} 정보", self.show_about, None),
            ("종료", self.close, QKeySequence.StandardKey.Quit),
        ]
        for a in acts:
            if a is None:
                menu.addSeparator()
                continue
            text, slot, key = a
            act = QAction(text, self)
            act.triggered.connect(slot)
            if key is not None:
                act.setShortcut(key)
                self.addAction(act)  # 메뉴를 열지 않아도 단축키 동작
            menu.addAction(act)
        self._menu = menu
        self.more_btn.clicked.connect(self._show_menu)
        if sys.platform == "darwin":  # macOS 는 화면 상단 메뉴 막대에도 표시
            mb = self.menuBar()
            fm = mb.addMenu("파일")
            fm.addActions(menu.actions())
        else:
            self.menuBar().hide()

    def _show_menu(self):
        from PySide6.QtCore import QPoint

        br = self.more_btn.mapToGlobal(self.more_btn.rect().bottomRight())
        self._menu.exec(QPoint(br.x() - self._menu.sizeHint().width(), br.y() + 6))

    # ========================================================= settings ===
    def _restore_settings(self):
        s = self.settings
        self.output_dir = Path(s.value("out_dir", str(paths.default_output_dir())))
        self.mode_ctl.set_current(str(s.value("mode", "4")))
        self.hq_box.setChecked(s.value("hq", False, type=bool))
        idx = self.device_combo.findData(s.value("device", "auto"))
        self.device_combo.setCurrentIndex(max(0, idx))
        idx = self.fmt_combo.findData(s.value("fmt", audio_io.DEFAULT_FORMAT))
        self.fmt_combo.setCurrentIndex(max(0, idx))
        self.mr_volume_box.setChecked(s.value("mr_volume", True, type=bool))
        idx = self.engine_combo.findData(s.value("engine", "demucs"))
        self.engine_combo.blockSignals(True)
        self.engine_combo.setCurrentIndex(max(0, idx))
        self.engine_combo.blockSignals(False)
        self.keep_native_rate = s.value("keep_native", True, type=bool)
        self.keep_drums_box.blockSignals(True)
        self.keep_drums_box.setChecked(s.value("keep_drums", True, type=bool))
        self.keep_drums_box.blockSignals(False)
        self._on_mode(self.mode_ctl.current(), save=False)

    def _save_settings(self):
        s = self.settings
        s.setValue("out_dir", str(self.output_dir))
        s.setValue("mode", self.mode_ctl.current())
        s.setValue("hq", self.hq_box.isChecked())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("fmt", self.fmt_combo.currentData())
        s.setValue("mr_volume", self.mr_volume_box.isChecked())
        s.setValue("engine", self.engine_key())
        s.setValue("keep_native", self.keep_native_rate)
        s.setValue("keep_drums", self.keep_drums_box.isChecked())
        s.sync()

    def engine_key(self) -> str:
        return self.engine_combo.currentData() or "demucs"

    def set_keep_native_rate(self, v: bool):
        self.keep_native_rate = bool(v)
        self._save_settings()

    # ============================================================ state ===
    def busy(self) -> bool:
        # 스레드 참조는 finished 처리(완료 시그널 처리 이후)에서 해제되므로 참조 존재 여부로 판단
        return any(t is not None for t in (self.load_thread, self.sep_thread, self.export_thread))

    def _thread_finished(self, attr: str):
        t = getattr(self, attr)
        setattr(self, attr, None)
        if t is not None:
            t.deleteLater()

    def _set_state(self):
        busy = self.busy()
        has_file = self.loaded is not None
        has_result = self.result is not None
        self.open_btn.setEnabled(not busy)
        self.start_btn.setEnabled(has_file and not busy)
        for b in self.mode_ctl.buttons.values():
            b.setEnabled(not busy)
        self.hq_box.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        for w in (self.mr_preview_btn, self.mr_save_btn, self.export_sel_btn, self.export_all_btn):
            w.setEnabled(has_result and not busy)
        for w in (self.play_btn, self.stop_btn, self.rew_btn, self.seek):
            w.setEnabled(has_file)
        self.chip_empty.setVisible(not has_result)
        self.start_btn.setText("다시 분리" if has_result else "AI 분리 시작")

    def _on_mode(self, key: str, save: bool = True):
        self.mode_ctl.set_current(key)
        eng = self.engine_key()
        title, text = MODE_INFO[(eng, key)]
        self.info_title.setText(title)
        self.info_text.setText(text)
        e = engine.ENGINES[eng]
        self.engine_hint.setText(e.note)
        self.engine_hint.setProperty("role", "warn" if eng == "roformer" else "dim")
        repolish(self.engine_hint)
        self._on_mode_options()
        self._update_device_hint()
        if save:
            self._save_settings()

    def _on_mode_options(self, *_):
        key = self.mode_ctl.current()
        hq = self.hq_box.isChecked()
        eng = self.engine_key()
        names = engine.required_models(eng, key, hq)
        if eng == "roformer":
            tip = "고품질: 구간을 4번 겹쳐 처리해 경계 잡음을 줄임 (처리 시간 약 2배)"
        elif key == "6":
            tip = ("고품질: 미세조정 모델 4개(htdemucs_ft)로 보컬·드럼·베이스를 분리하고, "
                   "기타·피아노는 2회 처리해 평균 (처리 시간 크게 증가)")
        else:
            tip = "고품질: 미세조정 모델 4개(htdemucs_ft) 사용 + 구간 겹침 증가 (처리 시간 약 4~8배)"
        self.hq_box.setToolTip(tip)
        missing = sum(models.missing_download_mb(n) for n in names)
        status = "설치됨" if not missing else f"첫 사용 시 약 {missing}MB 다운로드"
        self.hq_hint.setText(f"AI 모델 ({' + '.join(names)}) · {status}")
        self.hq_hint.setToolTip(tip)
        self.settings.setValue("hq", hq)

    def _on_device_pref(self, *_):
        self._update_device_hint()
        self.settings.setValue("device", self.device_combo.currentData())

    def _on_devices(self, devices: list):
        self.devices = devices
        gpus = [d for d in devices if d.key != "cpu"]
        item = self.device_combo.model().item(1)
        if item is not None:
            item.setEnabled(bool(gpus))
            self.device_combo.setItemText(1, f"GPU — {gpus[0].label}" if gpus else "GPU (사용 불가)")
        if not gpus and self.device_combo.currentData() == "gpu":
            self.device_combo.setCurrentIndex(0)
        self._update_device_hint()

    def _update_device_hint(self):
        if not self.devices:
            return
        pref = self.device_combo.currentData()
        gpus = [d for d in self.devices if d.key != "cpu"]
        cpu = self.devices[-1]
        if pref == "cpu" or not gpus:
            msg = f"{cpu.label}로 처리합니다." + ("" if gpus else " (사용 가능한 GPU 없음)")
            if self.engine_key() == "roformer":
                msg += "\n고급 엔진은 CPU에서 매우 느립니다 (곡 길이의 수 배~10배 이상)."
        else:
            msg = f"GPU 가속 사용: {gpus[0].label}\n문제가 생기면 자동으로 CPU로 다시 처리합니다."
        self.device_hint.setText(msg)

    def _resolve_device(self) -> str:
        pref = self.device_combo.currentData()
        if pref == "cpu":
            return "cpu"
        if not self.devices:
            return system.choose_device(pref).key
        gpus = [d for d in self.devices if d.key != "cpu"]
        return gpus[0].key if gpus else "cpu"

    def set_status(self, text: str, state: str = "ok"):
        """state: ok / busy / error"""
        self.status_msg.setText(text)
        self.status_msg.setToolTip(text)
        name, col, ic, txt = {
            "ok": ("StatusOk", theme.SUCCESS, "check-circle", "준비 완료"),
            "busy": ("StatusBusy", theme.ACCENT_2, "sparkles", "처리 중"),
            "error": ("StatusErr", theme.DANGER, "alert", "확인 필요"),
        }[state]
        self.status_icon.setPixmap(icons.pixmap(ic, col, 22))
        self.status_state.setText(txt)
        self.status_state.setObjectName(name)
        repolish(self.status_state)

    # ============================================================= file ===
    def browse_file(self):
        if self.busy():
            return
        exts = " ".join(f"*{e}" for e in sorted(audio_io.SUPPORTED_EXTENSIONS))
        start = self.settings.value("last_dir", str(Path.home() / "Music"))
        fn, _ = QFileDialog.getOpenFileName(self, "음악 파일 열기", start,
                                            f"오디오/동영상 ({exts});;모든 파일 (*)")
        if fn:
            self.settings.setValue("last_dir", str(Path(fn).parent))
            self.open_file(Path(fn))

    def open_file(self, path: Path):
        if self.busy():
            QMessageBox.information(self, APP_NAME, "작업이 끝난 뒤 다른 파일을 열 수 있습니다.")
            return
        path = Path(path)
        if path.suffix.lower() not in audio_io.SUPPORTED_EXTENSIONS:
            r = QMessageBox.question(self, APP_NAME,
                                     f"'{path.suffix}' 는 일반적인 오디오 형식이 아닙니다.\n그래도 열어볼까요?")
            if r != QMessageBox.StandardButton.Yes:
                return
        dur = audio_io.probe_duration(path)
        if dur:
            need = system.stem_bytes(system.samples_for(dur)) + 50 * system.MB
            if system.free_disk(self.session_dir) < need:
                self._error("임시 저장 공간이 부족합니다.\n"
                            f"필요: {system.human_bytes(need)}, 남은 공간: "
                            f"{system.human_bytes(system.free_disk(self.session_dir))}")
                return
            if dur > 20 * 60:
                r = QMessageBox.question(
                    self, APP_NAME,
                    f"곡 길이가 {audio_io.format_time(dur)} 로 매우 깁니다.\n"
                    "분리에 오랜 시간과 많은 메모리가 필요합니다. 계속할까요?")
                if r != QMessageBox.StandardButton.Yes:
                    return
        self._reset_tracks(clear_original=True)
        self._invalidate_key_cache()
        self._purge_cache()
        self.set_status(f"불러오는 중: {path.name}", "busy")
        self.file_title.setText("불러오는 중…")
        self.run_counter += 1
        self.load_thread = LoadThread(path, self.session_dir / f"original{self.run_counter}.npy", self)
        self.load_thread.done.connect(self._on_loaded)
        self.load_thread.failed.connect(self._on_load_failed)
        self.load_thread.finished.connect(lambda: (self._thread_finished("load_thread"), self._set_state()))
        self.load_thread.start()
        self._set_state()

    def _on_loaded(self, la: LoadedAudio):
        self.loaded = la
        fm = self.file_title.fontMetrics()
        rate = f" · {la.native_rate / 1000:g}kHz"
        self.file_title.setText(fm.elidedText(la.source.name, Qt.TextElideMode.ElideMiddle, 300 - fm.horizontalAdvance(rate)) + rate)
        self.file_title.setToolTip(str(la.source))
        self.lcd.len_lbl.setText(audio_io.format_time(la.seconds))
        self.ruler.set_duration(la.seconds)
        self.viewport.reset()
        self.stack.setCurrentIndex(1)
        orig = self._np_load(la.mix_path)
        self._add_row("original", "원본", "Original · 참고용", engine.stem_color("original"), la.peaks, True)
        self.mixer.set_tracks([Track("original", orig, reference=True)])
        self.mixer.seek(0)
        self.empty_hint.show()
        self.track_paths = {"original": la.mix_path}
        self._invalidate_key_cache()
        self.set_status(f"{la.source.name} ({audio_io.format_time(la.seconds)}) — 모드를 고르고 "
                        "[AI 분리 시작]을 누르세요.")
        self._set_state()

    def _on_load_failed(self, msg: str):
        self.file_title.setText("음악을 더 자유롭게, 당신의 사운드로")
        self.set_status("파일을 불러오지 못했습니다.", "error")
        if self.loaded is None:
            self.stack.setCurrentIndex(0)
        self._error(msg)

    @staticmethod
    def _np_load(path: Path):
        import numpy as np

        return np.load(path, mmap_mode="r")

    # ====================================================== separation ===
    def start_separation(self):
        if self.loaded is None or self.busy():
            return
        mode = self.mode_ctl.current()
        hq = self.hq_box.isChecked()
        eng = self.engine_key()
        model_name = os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or None
        if model_name and model_name.startswith("roformer"):
            eng = "roformer"
        interactive = not os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE")

        need = [n for n in engine.required_models(eng, mode, hq, model_name) if models.missing_download_mb(n)]
        if need:
            mb = sum(models.missing_download_mb(n) for n in need)
            free = system.free_disk(paths.user_models_dir())
            if free < mb * system.MB * 1.2:
                self._error(f"AI 모델을 저장할 공간이 부족합니다.\n필요: 약 {mb}MB, "
                            f"남은 공간: {system.human_bytes(free)}")
                return
            if interactive:
                extra = ""
                if "bs_roformer_sw" in need:
                    extra = ("\n\n고급 엔진(BS-RoFormer SW)은 커뮤니티에서 학습해 공개한 모델로, "
                             "모델 파일의 사용 조건이 명시되어 있지 않습니다. 개인 용도로 사용해 주세요.")
                r = QMessageBox.question(
                    self, "AI 모델 다운로드",
                    f"이 설정에는 AI 모델 {', '.join(need)} (약 {mb}MB)이 필요합니다.\n"
                    "처음 한 번만 인터넷에서 다운로드하며, 이후에는 인터넷 없이 사용할 수 있습니다." + extra +
                    "\n\n지금 다운로드할까요?")
                if r != QMessageBox.StandardButton.Yes:
                    return
        if eng == "roformer" and interactive and self._resolve_device() == "cpu" and \
                not self.settings.value("warned_roformer_cpu", False, type=bool):
            r = QMessageBox.question(
                self, "고급 엔진 안내",
                "GPU가 없어 CPU로 처리합니다. 고급 엔진은 계산량이 많아 CPU에서는 곡 길이의 수 배~10배 이상 "
                "걸릴 수 있습니다 (예: 4분 곡 → 20분 이상).\n\n계속할까요?")
            if r != QMessageBox.StandardButton.Yes:
                return
            self.settings.setValue("warned_roformer_cpu", True)
        if mode == "6" and eng == "demucs" and interactive and \
                not self.settings.value("warned_6s_v2", False, type=bool):
            QMessageBox.information(
                self, "6-Stem 안내",
                "표준 엔진의 6-Stem은 보컬·드럼·베이스를 먼저 고품질로 분리한 뒤, 남은 소리에서 "
                "기타·피아노를 나눕니다.\n\n"
                "• 기타: 비교적 잘 분리됩니다.\n"
                "• 피아노: 소리가 작게 나오거나 '나머지 악기'로 새는 경우가 있습니다.\n\n"
                "피아노가 중요하다면 'AI 엔진'에서 '고급 (BS-RoFormer)'을 선택해 보세요.")
            self.settings.setValue("warned_6s_v2", True)

        m = engine.MODES[mode]
        n_src = 6 if mode == "6" else 4
        main = engine.plan_models(eng, mode, hq, model_name).get("main", "")
        bag = len(models.MODELS[main].sigs) if main in models.MODELS else 1
        req = system.separation_requirement(self.loaded.n_samples, n_src, len(m.outputs), self.session_dir, bag)
        if not req.disk_ok:
            self._error("디스크 공간이 부족하여 분리할 수 없습니다.\n"
                        f"필요: {system.human_bytes(req.disk_needed)}, 남은 공간: "
                        f"{system.human_bytes(req.disk_available)}\n위치: {req.disk_path}")
            return
        if not req.ram_ok and interactive:
            r = QMessageBox.warning(
                self, "메모리 부족 경고",
                f"사용 가능한 메모리가 부족할 수 있습니다.\n"
                f"예상 필요: {system.human_bytes(req.ram_needed)}, 현재 여유: "
                f"{system.human_bytes(req.ram_available)}\n\n"
                "다른 프로그램을 종료하면 더 안정적입니다. 그래도 계속할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return

        # 이전 분리 결과는 새 분리가 성공할 때까지 유지 (취소·실패 시 그대로 사용 가능)
        self.run_counter += 1
        out_dir = self.session_dir / f"run{self.run_counter}"
        device = self._resolve_device()
        self.notice_lbl.hide()
        self.sep_thread = SeparationThread(self.loaded.mix_path, mode, hq, device, out_dir, model_name, eng, self)
        self.sep_thread.progress.connect(self._on_progress)
        self.sep_thread.notice.connect(self._on_notice)
        self.sep_thread.done.connect(self._on_separated)
        self.sep_thread.failed.connect(self._on_sep_failed)
        self.sep_thread.cancelled.connect(self._on_sep_cancelled)
        self.sep_thread.finished.connect(lambda: (self._thread_finished("sep_thread"), self._work_finished()))
        self._show_progress(True)
        self._on_progress(0.0, "준비 중…")
        self.sep_thread.start()
        self._set_state()
        self.set_status(f"{m.title} 분리 중 ({engine.ENGINES[eng].short})… 처리 중에도 프로그램을 계속 "
                        "사용할 수 있습니다.", "busy")

    def _show_progress(self, on: bool, cancellable: bool = True):
        self.progress.setVisible(on)
        self.progress_lbl.setVisible(on)
        self.cancel_btn.setVisible(on and cancellable)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("취소")
        self.start_btn.setVisible(not on)

    def _on_progress(self, frac: float, msg: str):
        self.progress.setValue(int(frac * 1000))
        self.progress_lbl.setText(msg)

    def _on_notice(self, msg: str):
        self.notice_lbl.setText(msg)
        self.notice_lbl.show()

    def cancel_work(self):
        for t in (self.sep_thread, self.export_thread):
            if t is not None and t.isRunning() and hasattr(t, "cancel"):
                t.cancel()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("취소 중…")
        self.progress_lbl.setText("현재 단계를 마무리하고 취소합니다…")

    def _work_finished(self):
        self._show_progress(False)
        self._set_state()
        self._on_mode_options()

    def _on_sep_cancelled(self):
        self.set_status("분리를 취소했습니다.")

    def _on_sep_failed(self, msg: str):
        self.set_status("분리에 실패했습니다.", "error")
        self._error(msg)

    def _on_separated(self, res: engine.SeparationResult):
        self._reset_tracks(clear_original=False)
        run_dir = res.stems[0].path.parent.name if res.stems else ""
        self._invalidate_key_cache()
        self._purge_cache(keep=self._source_files() | {run_dir})
        self.result = res
        tracks = [t for t in self.mixer.tracks if t.key == "original"]
        if tracks:
            tracks[0].mute = True
        for stem in res.stems:
            self._add_row(stem.key, engine.stem_label(stem.key, res.mode), engine.stem_english(stem.key),
                          engine.stem_color(stem.key), stem.peaks)
            tracks.append(Track(stem.key, self._np_load(stem.path)))
        self.mixer.set_tracks(tracks)
        if "original" in self.rows:
            self.rows["original"].set_muted(True)
        self.empty_hint.hide()
        self._build_chips(res)
        self._refresh_audible()
        self.track_paths = {"original": self.loaded.mix_path, **{st.key: st.path for st in res.stems}}
        dev = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(res.device, res.device)
        self.set_status(f"{engine.MODES[res.mode].title} 분리 완료 · {engine.ENGINES[res.engine].short} · {dev} · "
                        f"{audio_io.format_time(res.elapsed)} 소요 — ▶ 재생해서 들어 보세요.")
        self._set_state()
        if self.key_target != 0:
            self._apply_key()

    def _source_files(self) -> set:
        if not self.loaded:
            return set()
        names = {self.loaded.mix_path.name}
        if self.loaded.native_path:
            names.add(Path(self.loaded.native_path).name)
        return names

    # ============================================================ tracks ===
    def _add_row(self, key, name, sub, color, peaks, reference=False):
        from .icons import STEM_ICON

        number = len(self.rows) + 1
        row = TrackRow(self.viewport, key, number, name, sub, color, STEM_ICON.get(key, "wave"), peaks, reference)
        row.muteChanged.connect(self._on_mute)
        row.soloChanged.connect(self._on_solo)
        row.gainChanged.connect(self._on_gain)
        row.seekRequested.connect(self.seek_fraction)
        self.track_layout.insertWidget(self.track_layout.indexOf(self.empty_hint), row)
        self.rows[key] = row

    def _reset_tracks(self, clear_original: bool):
        self.player.pause()
        for key in list(self.rows):
            if key == "original" and not clear_original:
                continue
            row = self.rows.pop(key)
            row.setParent(None)
            row.deleteLater()
        if clear_original:
            self.mixer.clear()
            self.loaded = None
            self.track_paths = {}
            self.lcd.len_lbl.setText("00:00")
            self.ruler.set_duration(0)
        else:
            orig = [t for t in self.mixer.tracks if t.key == "original"]
            for t in orig:
                t.mute = False
            if "original" in self.rows:
                self.rows["original"].set_muted(False)
                self.rows["original"].set_solo(False)
            self.mixer.set_tracks(orig)
        self.result = None
        self._build_chips(None)
        self.empty_hint.show()
        gc.collect()

    def _purge_cache(self, keep: frozenset | set = frozenset()):
        """더 이상 쓰지 않는 임시 트랙 파일 삭제 (사용 중이라 지워지지 않는 파일은 종료 시 정리)."""
        for child in self.session_dir.iterdir():
            if child.name in keep:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError:
                pass

    def _track(self, key: str) -> Optional[Track]:
        return self.mixer.track(key)

    def _on_mute(self, key: str, v: bool):
        t = self._track(key)
        if t:
            t.mute = v
        self._refresh_audible()

    def _on_solo(self, key: str, v: bool):
        t = self._track(key)
        if t:
            t.solo = v
        self._refresh_audible()

    def _on_gain(self, key: str, g: float):
        t = self._track(key)
        if t:
            t.gain = g

    def _refresh_audible(self):
        tracks = self.mixer.tracks
        flags = audible_flags([t.mute for t in tracks], [t.solo for t in tracks])
        for t, f in zip(tracks, flags):
            if t.key in self.rows:
                self.rows[t.key].set_audible(f)

    def reset_mixer(self):
        for key, row in self.rows.items():
            row.set_solo(False)
            row.set_muted(key == "original" and self.result is not None)
        self._refresh_audible()

    # ============================================================ zoom ===
    def _on_viewport(self):
        vp = self.viewport
        span_units = int(vp.span * 10000)
        self.hscroll.blockSignals(True)
        self.hscroll.setRange(0, max(0, 10000 - span_units))
        self.hscroll.setPageStep(span_units)
        self.hscroll.setSingleStep(max(1, span_units // 20))
        self.hscroll.setValue(int(vp.start * 10000))
        self.hscroll.blockSignals(False)
        self.hscroll.setVisible(vp.span < 0.999)
        import math

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(math.log(vp.zoom) / math.log(vp.MAX_ZOOM) * 100))
        self.zoom_slider.blockSignals(False)

    def _zoom_slider_moved(self, v: int):
        self.viewport.set_zoom(self.viewport.MAX_ZOOM ** (v / 100), self._play_frac())

    def _play_frac(self) -> float:
        return self.mixer.position / self.mixer.length if self.mixer.length else 0.0

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ============================================================== MR ===
    def _build_chips(self, res: Optional[engine.SeparationResult]):
        from .icons import STEM_ICON

        for b in self.chips.values():
            b.setParent(None)
            b.deleteLater()
        self.chips.clear()
        if res is None:
            return
        cols = 2 if len(res.stems) <= 4 else 3
        for i, stem in enumerate(res.stems):
            b = ChipButton(f"{engine.stem_label(stem.key, res.mode)} 제거", STEM_ICON.get(stem.key, "wave"))
            b.setToolTip("누르면 MR에서 빠집니다 (빨간색 = 제외)")
            b.setChecked(stem.key == "vocals")
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.chip_grid.addWidget(b, i // cols, i % cols)
            self.chips[stem.key] = b

    def _mr_selection(self) -> tuple[list[str], list[str]]:
        excluded = [k for k, b in self.chips.items() if b.isChecked()]
        return [k for k in self.chips if k not in excluded], excluded

    def preview_mr(self):
        included, excluded = self._mr_selection()
        for key, row in self.rows.items():
            row.set_solo(False)
            row.set_muted(key == "original" or key in excluded)
        self._refresh_audible()
        if not self.player.playing:
            self.toggle_play()
        self.set_status("MR 미리 듣기: " + (", ".join(self._label(k) for k in excluded) or "없음") + " 제외")

    def export_mr(self):
        if self.result is None:
            return
        included, excluded = self._mr_selection()
        if not included:
            self._error("MR에 포함할 트랙이 없습니다. 제외할 악기를 줄여 주세요.")
            return
        use_vol = self.mr_volume_box.isChecked()
        gains = {k: (self.rows[k].gain if use_vol else 1.0) for k in included}
        fmt = self.fmt_combo.currentData()
        ext = audio_io.EXPORT_FORMATS[fmt][1]
        suffix = ("MR (" + ", ".join(self._label(k) for k in excluded) + " 제외)") if excluded else "전체 믹스"
        name = f"{self._song()} - {suffix}{self._key_suffix()}{ext}"
        path = audio_io.unique_path(self._song_dir() / audio_io.safe_filename(name))
        self._run_export([ExportJob(path, fmt, suffix, "mr", excluded=excluded, gains=gains)])

    # ========================================================== export ===
    def _song(self) -> str:
        return audio_io.safe_filename(self.loaded.source.stem if self.loaded else "track", 80)

    def _song_dir(self) -> Path:
        return self.output_dir / self._song()

    def _label(self, key: str) -> str:
        return engine.stem_label(key, self.result.mode if self.result else "4")

    def _key_suffix(self) -> str:
        return f" [Key {self.key_target:+d}]" if self.key_target else ""

    def _stem_jobs(self, keys: list[str]) -> list[ExportJob]:
        fmt = self.fmt_combo.currentData()
        ext = audio_io.EXPORT_FORMATS[fmt][1]
        jobs = []
        for k in keys:
            name = f"{self._song()} - {self._label(k)} ({engine.stem_english(k)}){self._key_suffix()}{ext}"
            path = audio_io.unique_path(self._song_dir() / audio_io.safe_filename(name))
            jobs.append(ExportJob(path, fmt, self._label(k), "stem", key=k))
        return jobs

    def export_selected(self):
        if self.result is None:
            return
        keys = [s.key for s in self.result.stems if self.rows[s.key].selected]
        if not keys:
            self._error("저장할 트랙의 스위치를 켜 주세요.")
            return
        self._run_export(self._stem_jobs(keys))

    def export_all(self):
        if self.result is None:
            return
        self._run_export(self._stem_jobs([s.key for s in self.result.stems]))

    def _export_source(self) -> ExportSource:
        res, la = self.result, self.loaded
        return ExportSource(stems={st.key: st.path for st in res.stems}, residual_key=res.stems[-1].key,
                            mix44_path=la.mix_path, native_path=la.native_path or la.mix_path,
                            native_rate=la.native_rate)

    def _run_export(self, jobs: list[ExportJob]):
        if self.busy() or not jobs or self.result is None:
            return
        rate = self.loaded.native_rate if self.keep_native_rate else audio_io.SAMPLE_RATE
        n = int(self.mixer.length * rate / audio_io.SAMPLE_RATE)
        need = sum(audio_io.estimate_export_bytes(n, j.fmt, rate) for j in jobs) + n * 8 * (len(jobs) + 7)
        free = system.free_disk(self._song_dir())
        if free < need:
            self._error(f"디스크 공간이 부족합니다.\n필요: {system.human_bytes(need)}, "
                        f"남은 공간: {system.human_bytes(free)}")
            return
        try:
            self._song_dir().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._error(f"저장 폴더를 만들 수 없습니다: {self._song_dir()}\n({exc})\n"
                        "[설정…]에서 다른 폴더를 선택해 주세요.")
            return
        opts = ExportOptions(keep_native_rate=self.keep_native_rate, semitones=self.key_target,
                             keep_drums_pitch=self.keep_drums_box.isChecked())
        self.export_thread = ExportThread(self._export_source(), jobs, opts, self.session_dir, self)
        self.export_thread.progress.connect(self._on_progress)
        self.export_thread.done.connect(self._on_exported)
        self.export_thread.failed.connect(lambda m: (self.set_status(m if "취소" in m else "저장 실패",
                                                                     "ok" if "취소" in m else "error"),
                                                     None if "취소" in m else self._error(m)))
        self.export_thread.finished.connect(
            lambda: (self._thread_finished("export_thread"), self._work_finished()))
        self._show_progress(True, cancellable=True)
        self.set_status("저장 중…", "busy")
        self.export_thread.start()
        self._set_state()

    def _on_exported(self, report):
        files = report.files
        extra = f" · {report.sample_rate / 1000:g}kHz"
        if report.gain_db < 0:
            extra += f" · 클리핑 방지 {report.gain_db:.1f}dB"
        self.set_status(f"{len(files)}개 파일 저장 완료{extra} → {self._song_dir()}")
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return
        box = QMessageBox(self)
        box.setWindowTitle("저장 완료")
        box.setText(f"{len(files)}개 파일을 저장했습니다.")
        info = "\n".join(Path(f).name for f in files[:8]) + ("\n…" if len(files) > 8 else "")
        info += f"\n\n샘플레이트: {report.sample_rate / 1000:g}kHz"
        if report.gain_db < 0:
            info += (f"\n소리가 0dBFS를 넘는 부분이 있어 잘림을 막기 위해 음량을 {abs(report.gain_db):.1f}dB "
                     "낮췄습니다. (원래 음량 그대로 저장하려면 'WAV 32-bit float' 형식을 선택하세요)")
        box.setInformativeText(info)
        open_btn = box.addButton("폴더 열기", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("닫기", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            paths.open_in_file_manager(self._song_dir())

    # ============================================================== key ===
    def _on_key_slider(self, v: int):
        self.key_target = int(v)
        self.key_value.setText("0" if v == 0 else f"{v:+d}")
        self.key_value.setStyleSheet("" if v == 0 else f"color:{theme.ACCENT};")
        self.key_timer.start()

    def _on_keep_drums(self, *_):
        self._save_settings()
        if self.key_target != 0 and "drums" in self.track_paths:
            self._invalidate_key_cache()
            self._apply_key()

    def _invalidate_key_cache(self):
        if self.key_thread is not None:
            self.key_thread.cancel()
        self.key_cache = {}
        self.key_semitones = 0

    def _apply_key(self):
        """재생 중인 트랙에 키 변경 적용 (캐시가 있으면 즉시, 없으면 백그라운드로 변환)."""
        if not self.track_paths:
            return
        target = self.key_target
        if target == 0 or target in self.key_cache:
            self._swap_tracks(target, self.key_cache.get(target, self.track_paths))
            return
        if self.key_thread is not None:
            self.key_thread.cancel()
        t = KeyThread(self.track_paths, target, self.keep_drums_box.isChecked(),
                      self.session_dir / "key", self)
        gen = id(self.track_paths)
        t.progress.connect(lambda f, m: self.set_status(m, "busy"))
        t.done.connect(lambda semis, paths_, gen=gen: self._on_key_done(semis, paths_, gen))
        t.failed.connect(lambda m: (self.set_status("키 변경 실패", "error"), self._error(m)))
        t.finished.connect(lambda t=t: self._key_thread_finished(t))
        self.key_thread = t
        t.start()

    def _key_thread_finished(self, t):
        if self.key_thread is t:
            self.key_thread = None
        t.deleteLater()

    def _on_key_done(self, semis: int, paths_: dict, gen: int):
        if gen != id(self.track_paths):   # 그 사이 새 파일/새 분리 결과로 바뀜
            return
        self.key_cache[semis] = paths_
        if semis == self.key_target:
            self._swap_tracks(semis, paths_)
        else:
            self._apply_key()

    def _swap_tracks(self, semis: int, paths_: dict):
        for t in self.mixer.tracks:
            p = paths_.get(t.key)
            if p is not None:
                t.data = self._np_load(p)
        self.mixer.set_tracks(list(self.mixer.tracks))
        self.key_semitones = semis
        label = pitch.key_label(semis)
        if not self.busy():
            self.set_status(f"키: {label}" + (" · 재생과 저장에 적용됩니다" if semis else ""))

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", str(self.output_dir))
        if d:
            self.output_dir = Path(d)
            self._save_settings()
            self.set_status(f"저장 폴더: {self.output_dir}")

    def open_output_dir(self):
        target = self._song_dir() if self.loaded and self._song_dir().exists() else self.output_dir
        target.mkdir(parents=True, exist_ok=True)
        paths.open_in_file_manager(target)

    # ========================================================= playback ===
    def toggle_play(self):
        if self.mixer.length == 0:
            return
        if self.player.playing:
            self.player.pause()
        elif not self.player.play():
            self._error(self.player.last_error or "재생할 수 없습니다.")
        self.play_btn.set_playing(self.player.playing)

    def stop(self):
        self.player.stop()
        self.play_btn.set_playing(False)

    def rewind(self):
        self.mixer.seek(0)
        self.viewport.ensure_visible(0.0)

    def seek_fraction(self, frac: float):
        self.mixer.seek(int(frac * self.mixer.length))

    def _seek_moved(self, v: int):
        if self._seeking:
            self.seek_fraction(v / 10000)

    def _seek_released(self):
        self._seeking = False
        self.seek_fraction(self.seek.value() / 10000)

    def _tick(self):
        self.play_btn.set_playing(self.player.playing)
        if self.mixer.length == 0:
            return
        sec = self.player.display_seconds()
        frac = min(1.0, sec / max(1e-6, self.mixer.duration))
        m, s = divmod(sec, 60)
        self.lcd.pos_lbl.setText(f"{int(m):02d}:{s:04.1f}")
        if not self._seeking:
            self.seek.blockSignals(True)
            self.seek.setValue(int(frac * 10000))
            self.seek.blockSignals(False)
        if self.player.playing:
            self.viewport.ensure_visible(frac)
        self.ruler.set_position(frac)
        for row in self.rows.values():
            row.wave.set_position(frac)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        wide = self.width() >= 1380  # 좁은 창에서는 마스터 볼륨을 숨겨 헤더가 겹치지 않게
        self.master.setVisible(wide)
        self.vol_icon.setVisible(wide)

    # ======================================================= drag & drop ===
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and not self.busy():
            e.acceptProposedAction()
            self.drop_zone.set_hover(True)

    def dragLeaveEvent(self, e):
        self.drop_zone.set_hover(False)

    def dropEvent(self, e):
        self.drop_zone.set_hover(False)
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_file(Path(urls[0].toLocalFile()))

    # ============================================================== misc ===
    def _error(self, msg: str):
        log.warning("error dialog: %s", msg)
        self.last_error = msg
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return
        QMessageBox.critical(self, APP_NAME, msg)

    def import_models(self):
        d = QFileDialog.getExistingDirectory(self, "가져올 models 폴더 선택 (htdemucs 등 하위 폴더 포함)")
        if not d:
            return
        try:
            done = models.copy_into_user_dir(Path(d))
        except OSError as exc:
            self._error(f"모델 복사 실패: {exc}")
            return
        if done:
            QMessageBox.information(self, APP_NAME, "가져온 모델: " + ", ".join(done))
        else:
            self._error("선택한 폴더에서 모델(htdemucs, htdemucs_ft, htdemucs_6s 폴더)을 찾지 못했습니다.")
        self._on_mode_options()

    def show_help(self):
        QMessageBox.information(self, "사용 방법", HELP_TEXT)

    def show_about(self):
        import platform

        try:
            import torch

            tv = torch.__version__
        except Exception:
            tv = "?"
        QMessageBox.about(
            self, f"{APP_NAME} 정보",
            f"<b>{APP_NAME}</b> v{__version__}<br><br>"
            "AI 음원 분리: Demucs (Meta AI Research, MIT License)<br>"
            "고급 엔진: BS-RoFormer (코드 MIT · SW 모델 by jarredou, 커뮤니티 모델)<br>"
            "키 조절: Signalsmith Stretch (MIT) · GUI: Qt for Python / PySide6 (LGPLv3) · "
            "글꼴: Pretendard (SIL OFL 1.1)<br>"
            "오디오 변환: FFmpeg · MP3 인코더: LAME<br>"
            f"PyTorch {tv} · Python {platform.python_version()} · {platform.system()} {platform.machine()}<br><br>"
            "분리된 음원의 저작권은 원곡 권리자에게 있습니다.")

    def closeEvent(self, e):
        if self.busy():
            r = QMessageBox.question(self, APP_NAME, "작업이 진행 중입니다. 취소하고 종료할까요?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.cancel_work()
            for t in (self.sep_thread, self.export_thread, self.load_thread):
                if t is not None and t.isRunning():
                    t.wait(15000)
        if self.key_thread is not None:
            self.key_thread.cancel()
            self.key_thread.wait(30000)
        if self.device_thread.isRunning():  # 시작 직후 바로 닫아도 스레드 정리 후 종료 (강제 종료 방지)
            self.device_thread.wait(15000)
        self._save_settings()
        self.timer.stop()
        self.player.shutdown()
        self.mixer.clear()
        self.result = None
        self.loaded = None
        gc.collect()
        shutil.rmtree(self.session_dir, ignore_errors=True)
        e.accept()


HELP_TEXT = """1. 음악 파일을 창에 끌어다 놓거나 [파일 열기]를 누릅니다.
2. 왼쪽에서 분리 모드와 AI 엔진을 고릅니다.
   • 2-Stem: 보컬 / 반주 · 4-Stem: 보컬 / 드럼 / 베이스 / 기타 악기
   • 6-Stem: 보컬 / 드럼 / 베이스 / 기타 / 피아노 / 나머지
   • 표준(Demucs): 빠르고 안정적 · 고급(BS-RoFormer): 피아노·기타가 훨씬 깔끔 (GPU 권장)
3. [AI 분리 시작]을 누릅니다.
4. 분리가 끝나면 ▶(Space)로 재생합니다.
   • M: 음소거 · S: 솔로(이 트랙만 듣기) · 슬라이더: 볼륨(더블클릭 = 100%)
   • 파형이나 시간 눈금을 클릭하면 그 위치로 이동합니다.
   • Ctrl(⌘)+휠 또는 오른쪽 위 −/+ 로 파형을 확대, Shift+휠로 좌우 이동
5. MR 만들기: 뺄 악기 버튼을 누르고(빨간색 = 제외) [MR 저장].
   원본에서 그 악기만 빼므로 남는 악기는 원본 음질 그대로입니다.
6. 키 조절: −/+ 또는 슬라이더로 반음 단위(−12~+12) 조절. 재생과 저장에 모두 적용됩니다.
7. 저장: 아래쪽에서 형식을 고르고 [선택한 트랙 저장](트랙 스위치 켠 것) 또는 [전체 트랙 저장].
   • 기본은 WAV 24-bit · 원본 샘플레이트. MP3도 선택할 수 있습니다.
   • 저장 폴더와 샘플레이트 설정은 [설정…]에서 바꿉니다.

• NVIDIA GPU 또는 Apple Silicon Mac에서는 GPU로 훨씬 빠르게 처리됩니다.
• GPU에서 문제가 생기면 자동으로 CPU로 다시 처리합니다.
• 처리 중 [취소]를 누르면 바로 멈춥니다."""
