"""메인 윈도우."""
from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QProcess, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QScrollBar, QSizePolicy, QSlider,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import (APP_BRAND, APP_NAME, APP_SHORT, __version__, audio_io, engine, models, paths, pipeline, pitch, system,
                youtube)
from ..dsp import audible_flags
from ..player import Mixer, Player, Track
from . import icons, theme
from .home import HomePage
from .widgets import (COLUMN_GAP, HEADER_WIDTH, ROW_SPACING, ChipButton, DropZone, ElideLabel, Lcd, LogoTile,
                      PlayButton, TimeRuler, TrackRow, TracksPanel, Viewport, glow, icon_button, label, repolish,
                      text_button)
from ..export import ExportJob, ExportOptions, ExportSource
from .workers import (ExportThread, KeyThread, LoadedAudio, LoadThread, PipelineThread, SeparationThread,
                      YouTubeLoadThread, YouTubeUpdateThread)

log = logging.getLogger(__name__)

MODE = "2"      # 보컬 담당 사용자용: 2-Stem(보컬 / 반주)만 사용

ENGINE_INFO = {
    "roformer": ("BS-RoFormer SW 로 보컬 분리",
                 "보컬과 반주 품질이 가장 좋은 모델입니다 (MVSEP 기준 보컬 SDR 11.30dB, Demucs 8.33dB). "
                 "반주는 '원본 − 보컬'로 만들어 악기 소리가 원본 그대로 남습니다."),
    "demucs": ("Demucs 로 빠르게 보컬 분리",
               "가볍고 빠른 모델입니다 (MVSEP 기준 보컬 SDR 8.33dB). CPU 만 있는 PC에서 시간을 줄이고 싶을 때 쓰세요. "
               "반주는 '원본 − 보컬'로 만들어 악기 소리가 원본 그대로 남습니다."),
}


class DeviceThread(QThread):
    done = Signal(list)

    def run(self):
        try:
            self.done.emit(system.detect_devices())
        except Exception as exc:
            log.exception("device detection failed")
            self.done.emit([system.DeviceInfo("cpu", f"CPU (장치 감지 실패: {exc})")])


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
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(10)
        lay.addWidget(label("저장 위치", "sectionLabel"))
        row = QHBoxLayout()
        self.path_lbl = label(str(win.output_dir), "dim", wrap=True)
        self.path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        change = text_button("변경…", "folder", None, 36)
        change.clicked.connect(self._change)
        row.addWidget(self.path_lbl, 1)
        row.addWidget(change)
        lay.addLayout(row)
        lay.addWidget(label("곡마다 하위 폴더가 만들어지고, 그 안에 결과 파일이 저장됩니다.", "faint", wrap=True))
        lay.addSpacing(6)
        lay.addWidget(label("출력 형식", "sectionLabel"))
        self.fmt = QComboBox()
        for key, (lbl, _) in audio_io.EXPORT_FORMATS.items():
            self.fmt.addItem(lbl, key)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(win.fmt_combo.currentData())))
        self.fmt.currentIndexChanged.connect(lambda *_: win.set_format(self.fmt.currentData()))
        lay.addWidget(self.fmt)
        lay.addWidget(label("WAV 24-bit(기본)·FLAC 은 무손실이라 추가 음질 손실이 없습니다. MP3 320kbps 는 마지막에 한 번만 "
                            "인코딩합니다. 무손실 형식으로 저장해도 원본(예: YouTube·MP3)에 없던 음질이 생기지는 않습니다.",
                            "faint", wrap=True))
        lay.addSpacing(6)
        lay.addWidget(label("음질", "sectionLabel"))
        self.native_box = QCheckBox("MR을 원본 샘플레이트로 저장 (권장)")
        self.native_box.setChecked(win.keep_native_rate)
        self.native_box.toggled.connect(win.set_keep_native_rate)
        lay.addWidget(self.native_box)
        lay.addWidget(label("AI 는 44.1kHz 에서 처리합니다. MR 은 원본(예: 48kHz)에서 보컬만 빼서 만들기 때문에 남는 악기는 "
                            "리샘플링 없이 원본 그대로입니다. 보컬은 AI 출력 그대로 44.1kHz 로 저장합니다(업샘플링 안 함). "
                            "0dBFS 를 넘는 경우에만 고정 배율로 음량을 살짝 낮추고, 리미터·EQ·노멀라이즈는 쓰지 않습니다.",
                            "faint", wrap=True))
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
        lay.addSpacing(6)
        lay.addWidget(label("YouTube 기능", "sectionLabel"))
        yrow = QHBoxLayout()
        self.yt_lbl = label(f"yt-dlp {youtube.ytdlp_version()} · JavaScript 엔진: "
                            f"{'Deno' if youtube.find_deno() else '없음'}", "dim", wrap=True)
        yrow.addWidget(self.yt_lbl, 1)
        ub = text_button("업데이트", "download", None, 36)
        ub.setToolTip("YouTube 가 바뀌어 받기가 안 될 때 최신 다운로드 도구(yt-dlp)를 받습니다.")
        ub.clicked.connect(win.update_youtube)
        yrow.addWidget(ub)
        lay.addLayout(yrow)
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
        self.setMinimumSize(*self.HOME_MIN)
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
        self.pipeline_thread: Optional[PipelineThread] = None
        self.yt_thread: Optional[YouTubeLoadThread] = None
        self.yt_update_thread: Optional[YouTubeUpdateThread] = None
        self.last_prep = None                  # 홈 작업: 마지막으로 디코딩한 입력 (재사용)
        self.last_sep = None
        self.last_sep_key = None
        self.last_job = None                   # 마지막 홈 작업 결과 (pipeline.JobResult)
        self.last_pipeline_error = None
        self._job_preset_override: Optional[str] = None
        self._job_t0 = 0.0

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._set_state()
        self.show_home()

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
    HOME_MIN = (440, 640)
    ADV_MIN = (1220, 760)

    def _build_ui(self):
        self.pages = QStackedWidget()
        self.home = HomePage()
        self.pages.addWidget(self.home)
        self.pages.addWidget(self._build_advanced())
        self.setCentralWidget(self.pages)
        self._wire_home()

    def show_home(self):
        self.pages.setCurrentIndex(0)
        self.setMinimumSize(*self.HOME_MIN)
        self._update_home_footer()

    def show_advanced(self):
        self.setMinimumSize(*self.ADV_MIN)
        if self.width() < self.ADV_MIN[0] or self.height() < self.ADV_MIN[1]:
            self.resize(max(self.width(), self.ADV_MIN[0]), max(self.height(), self.ADV_MIN[1]))
        self.pages.setCurrentIndex(1)

    def _build_advanced(self) -> QWidget:
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
        return root

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Header")
        bar.setFixedHeight(92)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(10)
        self.home_btn = text_button("홈", "chevron-left", None, 46, theme.TEXT_DIM)
        self.home_btn.setToolTip("홈 화면으로 (원클릭 추출)")
        self.home_btn.clicked.connect(self.show_home)
        lay.addWidget(self.home_btn)
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
        self.yt_btn = text_button("YouTube", "link", None, 46, theme.TEXT_DIM)
        self.yt_btn.setToolTip("YouTube 영상 주소로 오디오 불러오기")
        self.yt_btn.clicked.connect(self.open_youtube)
        lay.addWidget(self.yt_btn)
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
        c, cl = card("1", "AI 음원 분리 · 2-Stem", "VOCALS / INSTRUMENTAL")
        info = QFrame()
        info.setObjectName("InfoBox")
        il = QVBoxLayout(info)
        il.setContentsMargins(14, 12, 14, 12)
        il.setSpacing(6)
        ih = QHBoxLayout()
        ih.setSpacing(8)
        spark = QLabel()
        spark.setPixmap(icons.pixmap("sparkles", "#e0a800", 18))
        self.info_title = label("", "sectionLabel", wrap=True)
        ih.addWidget(spark)
        ih.addWidget(self.info_title, 1)
        self.info_text = label("", "dim", wrap=True)
        il.addLayout(ih)
        il.addWidget(self.info_text)
        cl.addWidget(info)

        cl.addWidget(label("처리 방식", "sectionLabel"))
        seg = QHBoxLayout()
        seg.setSpacing(8)
        self.preset_group = QButtonGroup(self)
        self.preset_btns: dict[str, QPushButton] = {}
        for key in ("speed", "quality", "best"):
            pr = engine.PRESETS[key]
            b = QPushButton(pr.title)
            b.setProperty("kind", "seg")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(pr.description)
            b.setFixedHeight(38)
            self.preset_group.addButton(b)
            self.preset_btns[key] = b
            b.clicked.connect(lambda _=False, k=key: self._on_preset(k))
            seg.addWidget(b)
        cl.addLayout(seg)
        self.preset_hint = label("", "faint", wrap=True)
        cl.addWidget(self.preset_hint)

        cl.addWidget(label("AI 모델", "sectionLabel"))
        self.engine_combo = QComboBox()
        for key, eng in engine.ENGINES.items():
            self.engine_combo.addItem(icons.icon("sparkles" if key != "demucs" else "layers", theme.TEXT_DIM, 18),
                                      eng.title, key)
        self.engine_combo.setIconSize(QSize(18, 18))
        self.engine_combo.currentIndexChanged.connect(lambda *_: self._on_mode(MODE))
        cl.addWidget(self.engine_combo)
        self.engine_hint = label("", "dim", wrap=True)
        cl.addWidget(self.engine_hint)
        self.model_hint = label("", "faint", wrap=True)
        cl.addWidget(self.model_hint)

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
        glow(self.start_btn, theme.ACCENT, 20, 60, 3)
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

        # ---- 2. MR / 보컬 저장 ----
        c, cl = card("2", "MR(반주) · 보컬 저장", "EXPORT")
        cl.addWidget(label("MR 은 원본에서 보컬만 빼서 만들기 때문에 남는 악기는 원본 음질 그대로입니다. "
                           "저장 형식과 위치는 아래 막대와 [설정]에서 바꿉니다.", "dim", wrap=True))
        grow = QHBoxLayout()
        grow.setSpacing(10)
        self.guide_box = QCheckBox("가이드 보컬 작게 남기기")
        self.guide_box.setToolTip("MR 에 원래 보컬을 작게 섞어 저장합니다 (연습용 가이드).")
        self.guide_box.toggled.connect(self._on_guide)
        self.guide_slider = QSlider(Qt.Orientation.Horizontal)
        self.guide_slider.setRange(5, 50)
        self.guide_slider.setValue(20)
        self.guide_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.guide_slider.valueChanged.connect(self._on_guide)
        self.guide_lbl = label("20%", "dim")
        self.guide_lbl.setFixedWidth(40)
        grow.addWidget(self.guide_box)
        grow.addWidget(self.guide_slider, 1)
        grow.addWidget(self.guide_lbl)
        cl.addLayout(grow)
        self.chip_box = QWidget()          # (이전 버전 호환: 2-Stem 에서는 악기 칩을 쓰지 않음)
        self.chip_grid = QGridLayout(self.chip_box)
        self.chip_box.hide()
        self.chip_empty = label("분리를 마치면 저장할 수 있습니다.", "faint", wrap=True)
        cl.addWidget(self.chip_empty)
        mr = QHBoxLayout()
        mr.setSpacing(10)
        self.mr_preview_btn = text_button("MR 미리 듣기", "play", None, 46)
        self.mr_preview_btn.clicked.connect(self.preview_mr)
        self.mr_save_btn = text_button("MR 저장", "music", "primary", 46)
        self.mr_save_btn.clicked.connect(self.export_mr)
        glow(self.mr_save_btn, theme.ACCENT, 20, 60, 3)
        mr.addWidget(self.mr_preview_btn, 1)
        mr.addWidget(self.mr_save_btn, 1)
        cl.addLayout(mr)
        self.vocal_save_btn = text_button("보컬만 저장", "mic", None, 42, theme.TEXT_DIM)
        self.vocal_save_btn.clicked.connect(self.export_vocals)
        cl.addWidget(self.vocal_save_btn)
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
        self.formant_box = QCheckBox("보컬 음색(포먼트) 유지")
        self.formant_box.setToolTip("키를 바꿀 때 보컬 목소리가 가늘어지거나 굵어지지 않도록 음색을 유지합니다.\n"
                                    "보컬 트랙·보컬 결과물에만 적용되고 MR 에는 적용하지 않습니다.")
        self.formant_box.toggled.connect(self._on_formant)
        cl.addWidget(self.formant_box)
        self.key_hint = label("템포와 곡 길이는 그대로, 음높이만 바뀝니다. 0 이면 키 변경 처리를 하지 않습니다. "
                              "재생과 저장에 모두 적용됩니다.", "faint", wrap=True)
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
        self.status_msg = ElideLabel("음악 파일을 열어 시작하세요.")
        self.status_msg.setProperty("role", "dim")
        lay.addWidget(self.status_icon)
        lay.addWidget(self.status_state)
        lay.addWidget(sep)
        lay.addWidget(self.status_msg, 1)

        self.settings_btn = icon_button("settings", "설정 (저장 위치·출력 형식·샘플레이트)", 44, 19, "square",
                                        theme.TEXT_DIM)
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
            ("홈 화면", self.show_home, None),
            ("파일 열기…", self.browse_file, QKeySequence.StandardKey.Open),
            ("YouTube에서 불러오기…", self.open_youtube, None),
            ("저장 폴더 변경…", self.choose_output_dir, None),
            None,
            ("모든 음소거·솔로 해제", self.reset_mixer, None),
            ("전체 화면", self.toggle_fullscreen, QKeySequence.StandardKey.FullScreen),
            None,
            ("사용 방법", self.show_help, QKeySequence.StandardKey.HelpContents),
            ("AI 모델 폴더 열기", lambda: paths.open_in_file_manager(paths.user_models_dir()), None),
            ("AI 모델 가져오기 (오프라인 설치)…", self.import_models, None),
            ("YouTube 기능 업데이트", self.update_youtube, None),
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
        default = engine.app_default_preset()          # CPU 버전 = 속도 우선, CUDA 버전 = 품질 우선
        preset = str(s.value("preset2", default))      # (v1.2.0 의 'preset' 저장값은 쓰지 않음)
        self.preset_btns.get(preset, self.preset_btns[default]).setChecked(True)
        idx = self.device_combo.findData(s.value("device", "auto"))
        self.device_combo.setCurrentIndex(max(0, idx))
        idx = self.fmt_combo.findData(s.value("fmt", audio_io.DEFAULT_FORMAT))
        self.fmt_combo.setCurrentIndex(max(0, idx))
        idx = self.engine_combo.findData(s.value("engine2", "auto"))     # v1.2 부터 기본 '자동'
        self.engine_combo.blockSignals(True)
        self.engine_combo.setCurrentIndex(max(0, idx))
        self.engine_combo.blockSignals(False)
        self.keep_native_rate = s.value("keep_native", True, type=bool)
        self.formant_box.blockSignals(True)
        self.formant_box.setChecked(s.value("formant", True, type=bool))
        self.formant_box.blockSignals(False)
        self.guide_box.blockSignals(True)
        self.guide_box.setChecked(s.value("guide", False, type=bool))
        self.guide_box.blockSignals(False)
        self.guide_slider.setValue(int(s.value("guide_level", 20)))
        self.home.set_target(str(s.value("target", "mr")))
        self._on_guide()
        self._on_mode(MODE, save=False)
        self._update_home_footer()

    def _save_settings(self):
        s = self.settings
        s.setValue("out_dir", str(self.output_dir))
        s.setValue("preset2", self.preset_key())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("fmt", self.fmt_combo.currentData())
        s.setValue("engine2", self.engine_key())
        s.setValue("keep_native", self.keep_native_rate)
        s.setValue("formant", self.formant_box.isChecked())
        s.setValue("guide", self.guide_box.isChecked())
        s.setValue("guide_level", self.guide_slider.value())
        s.setValue("target", self.home.target())
        s.sync()

    def engine_key(self) -> str:
        return self.engine_combo.currentData() or "auto"

    def preset_key(self) -> str:
        for k, b in self.preset_btns.items():
            if b.isChecked():
                return k
        return engine.app_default_preset()

    def resolved_engine(self, preset: Optional[str] = None) -> str:
        return engine.resolve_engine(self.engine_key(), preset or self.preset_key(),
                                     os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or None)

    def set_keep_native_rate(self, v: bool):
        self.keep_native_rate = bool(v)
        self._save_settings()

    def set_format(self, fmt: str):
        idx = self.fmt_combo.findData(fmt)
        if idx >= 0:
            self.fmt_combo.setCurrentIndex(idx)
        self._save_settings()
        self._update_home_footer()

    # ============================================================ state ===
    def busy(self) -> bool:
        # 스레드 참조는 finished 처리(완료 시그널 처리 이후)에서 해제되므로 참조 존재 여부로 판단
        return any(t is not None for t in (self.load_thread, self.sep_thread, self.export_thread,
                                            self.pipeline_thread, self.yt_thread))

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
        self.yt_btn.setEnabled(not busy)
        self.start_btn.setEnabled(has_file and not busy)
        for b in self.preset_btns.values():
            b.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        for w in (self.mr_preview_btn, self.mr_save_btn, self.vocal_save_btn, self.export_sel_btn,
                  self.export_all_btn):
            w.setEnabled(has_result and not busy)
        for w in (self.play_btn, self.stop_btn, self.rew_btn, self.seek):
            w.setEnabled(has_file)
        self.chip_empty.setVisible(not has_result)
        self.start_btn.setText("다시 분리" if has_result else "AI 분리 시작")
        self.home_btn.setEnabled(True)

    def _on_preset(self, key: str):
        self.preset_btns[key].setChecked(True)
        self._on_mode(MODE)

    def _on_mode(self, key: str = MODE, save: bool = True):
        eng = self.resolved_engine()
        title, text = ENGINE_INFO[eng]
        self.info_title.setText(title)
        self.info_text.setText(text)
        e = engine.ENGINES[self.engine_key()]
        self.engine_hint.setText(e.note)
        self.engine_hint.setProperty("role", "warn" if self.engine_key() == "roformer" else "dim")
        repolish(self.engine_hint)
        self.preset_hint.setText(engine.PRESETS[self.preset_key()].description)
        self._on_mode_options()
        self._update_device_hint()
        if save:
            self._save_settings()

    def _on_mode_options(self, *_):
        names = engine.required_models(self.engine_key(), MODE, self.preset_key(),
                                       os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or None)
        missing = sum(models.missing_download_mb(n) for n in names)
        status = "설치됨" if not missing else f"첫 사용 시 약 {missing}MB 다운로드"
        self.model_hint.setText(f"사용 모델: {' + '.join(names)} · {status}")

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
            if self.resolved_engine() == "roformer":
                msg += ("\nBS-RoFormer 는 CPU 에서 곡 길이의 수 배~10배 이상 걸립니다. 빨리 받고 싶으면 "
                        "'속도 우선'을 선택하세요.")
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
        preset = self.preset_key()
        eng = self.engine_key()
        model_name = os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or None
        resolved = engine.resolve_engine(eng, preset, model_name)
        interactive = not os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE")

        need = [n for n in engine.required_models(eng, MODE, preset, model_name) if models.missing_download_mb(n)]
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
                    extra = ("\n\nBS-RoFormer SW 는 커뮤니티에서 학습해 공개한 모델로, 모델 파일의 사용 조건이 "
                             "명시되어 있지 않습니다. 개인 용도로 사용해 주세요.")
                r = QMessageBox.question(
                    self, "AI 모델 다운로드",
                    f"이 설정에는 AI 모델 {', '.join(need)} (약 {mb}MB)이 필요합니다.\n"
                    "처음 한 번만 인터넷에서 다운로드하며, 이후에는 인터넷 없이 사용할 수 있습니다." + extra +
                    "\n\n지금 다운로드할까요?")
                if r != QMessageBox.StandardButton.Yes:
                    return
        if resolved == "roformer" and interactive and self._resolve_device() == "cpu" and \
                not self.settings.value("warned_roformer_cpu", False, type=bool):
            r = QMessageBox.question(
                self, "처리 시간 안내",
                "GPU가 없어 CPU로 처리합니다. BS-RoFormer 는 계산량이 많아 CPU에서는 곡 길이의 수 배~10배 이상 "
                "걸릴 수 있습니다 (예: 4분 곡 → 15~40분).\n\n빨리 받으려면 '속도 우선'을 선택하세요. 계속할까요?")
            if r != QMessageBox.StandardButton.Yes:
                return
            self.settings.setValue("warned_roformer_cpu", True)

        m = engine.MODES[MODE]
        main = engine.plan_models(eng, MODE, preset, model_name).get("main", "")
        bag = len(models.MODELS[main].sigs) if main in models.MODELS else 1
        req = system.separation_requirement(self.loaded.n_samples, 6 if resolved == "roformer" else 4,
                                            len(m.outputs), self.session_dir, bag)
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
        self.sep_thread = SeparationThread(self.loaded.mix_path, MODE, preset, device, out_dir, model_name, eng, self)
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
        self.set_status(f"2-Stem 분리 중 ({engine.ENGINES[resolved].short}, {engine.PRESETS[preset].title})… "
                        "처리 중에도 프로그램을 계속 사용할 수 있습니다.", "busy")

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
        for t in (self.sep_thread, self.export_thread, self.yt_thread):
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
        self._purge_cache(keep=self._source_files() | {run_dir} | self._top_dirs([s.path for s in res.stems]))
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
        if self.loaded is not None and res.mode == MODE:    # 홈 화면 작업에서도 재사용
            self.last_sep = res
            self.last_sep_key = pipeline.separation_key(self.loaded.source_id, res.model, res.preset)
        self._set_state()
        if self.key_target != 0:
            self._apply_key()

    def _source_files(self) -> set:
        if not self.loaded:
            return set()
        return self._top_dirs([self.loaded.mix_path, self.loaded.native_path])

    def _top_dirs(self, files) -> set:
        """세션 폴더 바로 아래에서 이 파일들을 담고 있는 항목 이름 (정리할 때 지우지 않을 것)."""
        out = set()
        for f in files:
            if not f:
                continue
            try:
                out.add(Path(f).resolve().relative_to(self.session_dir.resolve()).parts[0])
            except (ValueError, IndexError):
                pass
        return out

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
        """2-Stem 에서는 악기 칩 대신 [MR 저장]/[보컬만 저장] 버튼을 사용."""
        for b in self.chips.values():
            b.setParent(None)
            b.deleteLater()
        self.chips.clear()

    def _on_guide(self, *_):
        on = self.guide_box.isChecked()
        self.guide_slider.setEnabled(on)
        self.guide_lbl.setText(f"{self.guide_slider.value()}%")
        self.guide_lbl.setEnabled(on)
        self.mr_save_btn.setText("MR 저장 (가이드 보컬 포함)" if on else "MR 저장")
        self.settings.setValue("guide", on)
        self.settings.setValue("guide_level", self.guide_slider.value())

    def _guide_gain(self) -> float:
        return self.guide_slider.value() / 100.0 if self.guide_box.isChecked() else 0.0

    def preview_mr(self):
        g = self._guide_gain()
        for key, row in self.rows.items():
            row.set_solo(False)
            if key == "vocals":
                row.set_muted(g == 0)
                if g:
                    row.slider.setValue(int(round(g * 100)))
            else:
                row.set_muted(key == "original")
        self._refresh_audible()
        if not self.player.playing:
            self.toggle_play()
        self.set_status("MR 미리 듣기" + (f" (가이드 보컬 {int(g * 100)}%)" if g else " (보컬 제거)"))

    def export_mr(self):
        if self.result is None:
            return
        g = self._guide_gain()
        fmt = self.fmt_combo.currentData()
        ext = audio_io.EXPORT_FORMATS[fmt][1]
        if g:
            suffix = f"MR (가이드 보컬 {int(round(g * 100))}%)"
            job = ExportJob(Path(), fmt, suffix, "mr", excluded=[], gains={"vocals": g})
        else:
            suffix = "MR (보컬 제거)"
            job = ExportJob(Path(), fmt, suffix, "mr", excluded=["vocals"], gains={})
        name = f"{self._song()} - {suffix}{self._key_suffix()}{ext}"
        job.path = audio_io.unique_path(self._song_dir() / audio_io.safe_filename(name))
        self._run_export([job])

    def export_vocals(self):
        if self.result is None:
            return
        self._run_export(self._stem_jobs(["vocals"]))

    # ========================================================== export ===
    def _song(self) -> str:
        return audio_io.safe_filename(self.loaded.name if self.loaded else "track", 60)

    def _song_dir(self) -> Path:
        return self.output_dir / self._song()

    def _label(self, key: str) -> str:
        return engine.stem_label(key, self.result.mode if self.result else MODE)

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
                            native_rate=la.native_rate, model_rate=res.model_rate)

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
                             keep_drums_pitch=True, formant_vocals=self.formant_box.isChecked())
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
        rates = sorted({d.sample_rate for d in report.details}) or [report.sample_rate]
        extra = " · " + "/".join(f"{r / 1000:g}kHz" for r in rates)
        if report.gain_db < 0:
            extra += f" · 클리핑 방지 {report.gain_db:.1f}dB"
        self.set_status(f"{len(files)}개 파일 저장 완료{extra} → {self._song_dir()}")
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return
        box = QMessageBox(self)
        box.setWindowTitle("저장 완료")
        box.setText(f"{len(files)}개 파일을 저장했습니다.")
        info = "\n".join(Path(f).name for f in files[:8]) + ("\n…" if len(files) > 8 else "")
        info += "\n\n" + "\n".join(f"{Path(d.path).name}: {d.verified}" for d in report.details[:4])
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
        self.key_value.setStyleSheet("" if v == 0 else f"color:{theme.ACCENT_2};")
        self.home.set_key(int(v))
        if self.track_paths:
            self.key_timer.start()

    def _on_home_key(self, v: int):
        self.key_slider.blockSignals(True)
        self.key_slider.setValue(int(v))
        self.key_slider.blockSignals(False)
        self.key_target = int(v)
        self.key_value.setText("0" if v == 0 else f"{v:+d}")
        self.key_value.setStyleSheet("" if v == 0 else f"color:{theme.ACCENT_2};")
        if self.track_paths:
            self.key_timer.start()

    def _on_formant(self, *_):
        self._save_settings()
        if self.key_target != 0 and "vocals" in self.track_paths:
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
        t = KeyThread(self.track_paths, target, True, self.session_dir / "key", self,
                      formant_vocals=self.formant_box.isChecked())
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
            self._update_home_footer()

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

    # ====================================================== 홈 화면 연결 ===
    def _wire_home(self):
        h = self.home
        h.advancedRequested.connect(self.show_advanced)
        h.settingsRequested.connect(lambda: SettingsDialog(self).exec())
        h.browseRequested.connect(self.home_browse)
        h.fileChosen.connect(self.home_set_file)
        h.startRequested.connect(self.start_home_job)
        h.keyChanged.connect(self._on_home_key)
        h.targetChanged.connect(lambda _t: self.settings.setValue("target", _t))
        h.status.cancelRequested.connect(self.cancel_home_job)
        h.status.fastRequested.connect(lambda: self.restart_home_job("speed"))
        h.result.openFolder.connect(self._open_job_folder)
        h.result.openFile.connect(self._open_job_file)
        h.result.showAdvanced.connect(self.show_advanced)
        h.result.newJob.connect(self._home_new_job)
        h.error.retry.connect(lambda: self.start_home_job())
        h.error.fast.connect(lambda: self.start_home_job(preset_override="speed"))
        h.error.update_youtube.connect(self.update_youtube)
        h.error.dismiss.connect(h.reset_panels)
        self.fmt_combo.currentIndexChanged.connect(lambda *_: (self._save_settings(), self._update_home_footer()))

    def _update_home_footer(self):
        if not hasattr(self, "output_dir"):
            return
        fmt = self.fmt_combo.currentData() or audio_io.DEFAULT_FORMAT
        self.home.set_footer(audio_io.EXPORT_FORMATS[fmt][0].replace(" (권장)", ""), self.output_dir)

    def home_browse(self):
        exts = " ".join(f"*{e}" for e in sorted(audio_io.SUPPORTED_EXTENSIONS))
        start = self.settings.value("last_dir", str(Path.home() / "Music"))
        fn, _ = QFileDialog.getOpenFileName(self, "음원 파일 선택", start, f"오디오/동영상 ({exts});;모든 파일 (*)")
        if fn:
            self.settings.setValue("last_dir", str(Path(fn).parent))
            self.home_set_file(Path(fn))

    def home_set_file(self, path: Path):
        path = Path(path)
        if self.busy():
            return
        if not path.is_file():
            self.home.show_error("파일을 열 수 없습니다", f"파일을 찾을 수 없습니다:\n{path}", False, False, False)
            return
        self.home.set_file(path, "")
        info = audio_io.probe_audio(path)
        if not info.codec:
            self.home.set_file_detail("오디오를 찾을 수 없는 파일입니다. 다른 파일을 선택해 주세요.")
            return
        dur = audio_io.format_time(info.duration) if info.duration else "?"
        detail = f"{info.summary()} · {dur}"
        if info.duration:
            try:
                dev = self._resolve_device()
                est = pipeline.estimate_seconds(int(info.duration * engine.MODEL_RATE), self.resolved_engine(), dev,
                                                self.preset_key())
                if dev == "cpu" and est >= 120:
                    detail += f" · 예상 처리 시간 약 {max(2, round(est / 60))}분 (그래픽카드 가속 없음)"
            except Exception:  # noqa: BLE001 - 표시용 추정
                log.debug("estimate failed", exc_info=True)
        self.home.set_file_detail(detail)

    def _home_steps(self, kind: str, semitones: int) -> list:
        steps = [("check", "입력 확인")]
        if kind == "youtube":
            steps.append(("acquire", "오디오 가져오기"))
        steps += [("decode", "디코딩"), ("model", "AI 모델 준비"), ("separate", "AI 음원 분리")]
        if semitones:
            steps.append(("key", "키 조절"))
        steps.append(("encode", "저장"))
        return steps

    def start_home_job(self, preset_override: Optional[str] = None):
        if self.busy():
            return
        h = self.home
        kind = h.source_kind()
        if kind not in ("file", "youtube"):
            h.show_error("입력을 확인해 주세요", "음원 파일을 선택하거나 YouTube 영상 주소를 입력해 주세요."
                         if kind == "" else "올바른 YouTube 영상 주소가 아닙니다.\n예: https://www.youtube.com/watch?v=…",
                         False, False, False)
            return
        if kind == "youtube" and not self._youtube_consent():
            return
        preset = preset_override or self.preset_key()
        self._job_preset_override = preset_override
        device = self._resolve_device()
        model_override = os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or None
        spec = pipeline.JobSpec(
            source=h.source_text(), out_dir=self.output_dir, target=h.target(), semitones=self.key_target,
            fmt=self.fmt_combo.currentData() or audio_io.DEFAULT_FORMAT, preset=preset, engine=self.engine_key(),
            device=device, formant_vocals=self.formant_box.isChecked(), keep_native_rate=self.keep_native_rate,
            model_override=model_override)
        self._save_settings()
        # 같은 입력이면 디코딩·분리 결과를 재사용 (결과물/키만 바꿔 다시 받을 때 빠름)
        prev_prep = self.last_prep if self.last_prep is not None else None
        self.player.pause()
        self.pipeline_thread = PipelineThread(spec, self.session_dir / "jobs", prev_prep, self.last_sep,
                                              self.last_sep_key, self)
        t = self.pipeline_thread
        t.progress.connect(self._on_home_progress)
        t.notice.connect(self.home.status.show_notice)
        t.done.connect(self._on_home_done)
        t.failed.connect(self._on_home_failed)
        t.cancelled.connect(self._on_home_cancelled)
        t.finished.connect(lambda: (self._thread_finished("pipeline_thread"), self._home_finished()))
        h.status.set_steps(self._home_steps(kind, spec.semitones))
        resolved = engine.resolve_engine(spec.engine, preset, model_override)
        h.status.fast_btn.setVisible(resolved == "roformer" and device == "cpu")
        h.status.set_progress(0.0, "준비 중…", self._job_subtitle(spec, resolved, device))
        h.set_busy(True)
        self._job_t0 = __import__("time").monotonic()
        self._job_spec = spec
        self._job_engine = resolved
        t.start()
        self._set_state()
        self.set_status("홈 화면에서 음원 추출 중…", "busy")

    def _job_subtitle(self, spec, resolved: str, device: str) -> str:
        dev = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(device, device)
        txt = f"{engine.ENGINES[resolved].short} · {engine.PRESETS[spec.preset].title} · {dev}"
        if resolved == "roformer" and device == "cpu":
            txt += " · CPU 에서는 시간이 오래 걸립니다 (곡 길이의 수 배)"
        return txt

    def restart_home_job(self, preset: str):
        if self.pipeline_thread is not None:
            self._restart_after_cancel = preset
            self.pipeline_thread.cancel()

    def cancel_home_job(self):
        if self.pipeline_thread is not None:
            self.pipeline_thread.cancel()
            self.home.status.cancel_btn.setEnabled(False)
            self.home.status.cancel_btn.setText("취소 중…")
            self.home.status.set_progress(self.home.status.bar.value() / 1000, "현재 단계를 마무리하고 취소합니다…")

    def _on_home_progress(self, stage: str, overall: float, msg: str):
        import time as _t

        st = self.home.status
        if stage != "done":
            st.set_stage(stage)
        el = _t.monotonic() - self._job_t0
        sub = f"경과 {audio_io.format_time(el)}"
        if overall > 0.08 and el > 5:
            remain = el / overall * (1 - overall)
            sub += f" · 남은 시간 약 {audio_io.format_time(remain)}"
        st.set_progress(overall, msg, self._job_subtitle(self._job_spec, self._job_engine, self._job_spec.device)
                        + " · " + sub)

    def _on_home_done(self, res):
        self.last_job = res
        self.last_prep = res.prepared
        self.last_sep = res.separation
        self.last_sep_key = getattr(res, "separation_key", None)
        self.home.status.finish_all()
        exp = res.export
        lines = []
        for d in exp.details:
            name = Path(d.path).name
            fmt = audio_io.EXPORT_FORMATS[d.fmt][0].replace(" (권장)", "")
            key = f" · 키 {d.semitones:+d}" + (" (음색 유지)" if d.formant else "") if d.semitones else ""
            lines.append(f"<b>{name}</b><br><span style='color:{theme.TEXT_DIM}'>{fmt} · {d.sample_rate / 1000:g}kHz · "
                         f"{audio_io.format_time(d.seconds)}{key}</span>")
        rep = res.report
        detail = []
        inp = rep["input"]
        if "youtube" in rep:
            y = rep["youtube"]
            detail.append(f"입력: YouTube {y['codec_name']} {y['abr_kbps']:g}kbps · {y['sample_rate'] / 1000:g}kHz "
                          f"(오디오 전용 스트림, 재인코딩 없음)")
        else:
            detail.append(f"입력: {inp['codec'].upper()} · {inp['sample_rate'] / 1000:g}kHz"
                          + (f" · {inp['bit_rate_kbps']}kbps" if inp['bit_rate_kbps'] and not inp['lossless'] else ""))
        sep = rep["separation"]
        dev = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(sep["device"], sep["device"])
        detail.append(f"분리: {engine.ENGINES[sep['engine']].short} ({engine.PRESETS[sep['preset']].title}) · {dev}"
                      + (" · 이전 결과 재사용" if sep["reused"] else f" · {audio_io.format_time(sep['elapsed_sec'])}"))
        steps = rep["decode"]["resample_steps"]
        lossy = rep.get("lossy_encodes", 0)
        detail.append("처리: float32 무손실 중간 데이터 · "
                      + (f"리샘플링 1회 ({steps[0].split(': ', 1)[-1]})" if steps else "리샘플링 없음")
                      + (f" · 손실 압축은 최종 MP3 저장 {lossy}회뿐" if lossy else " · 손실 압축 없음"))
        if exp.gain_db < 0:
            detail.append(f"0dBFS 를 넘는 부분이 있어 음량을 {abs(exp.gain_db):.1f}dB 낮췄습니다 (리미터 없이 고정 배율).")
        title = {"mr": "반주(MR) 파일을 저장했습니다", "vocals": "보컬 파일을 저장했습니다"}.get(
            self._job_spec.target, "파일을 저장했습니다")
        self.home.show_result(title, "<br>".join(lines), "\n".join(detail))
        self._adopt_pipeline_result(res)
        self.set_status(f"홈 작업 완료 → {Path(exp.files[0]).parent if exp.files else self.output_dir}")
        log.info("home job report: %s", rep)

    def _on_home_failed(self, err):
        self.last_pipeline_error = err
        self.last_error = str(err)
        log.warning("home job failed: %s", err)
        stage = pipeline.STAGE_LABEL.get(getattr(err, "stage", ""), "")
        title = f"{stage} 단계에서 문제가 생겼습니다" if stage else "처리하지 못했습니다"
        fast = bool(getattr(err, "can_use_fast_model", False)) and self._job_spec.preset != "speed"
        self.home.show_error(title, str(err), bool(getattr(err, "retryable", True)), fast,
                             bool(getattr(err, "can_update_youtube", False)))
        self.set_status("홈 작업 실패", "error")

    def _on_home_cancelled(self):
        self.set_status("작업을 취소했습니다.")

    def _home_finished(self):
        self.home.set_busy(False)
        self._set_state()
        preset = getattr(self, "_restart_after_cancel", None)
        if preset:
            self._restart_after_cancel = None
            QTimer.singleShot(100, lambda: self.start_home_job(preset_override=preset))

    def _home_new_job(self):
        self.home.reset_panels()
        self.home.url.clear()
        self.home._clear_file()

    def _open_job_folder(self):
        if self.last_job and self.last_job.files:
            paths.open_in_file_manager(Path(self.last_job.files[0]).parent)
        else:
            self.open_output_dir()

    def _open_job_file(self):
        if self.last_job and self.last_job.files:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_job.files[0])))

    def _adopt_pipeline_result(self, res):
        """홈 작업 결과를 고급 옵션 화면에 불러옴 (재생·다른 형식 저장용)."""
        prep = res.prepared
        try:
            self._reset_tracks(clear_original=True)
            self._invalidate_key_cache()
            la = LoadedAudio(prep.source_path, prep.mix44_path, prep.n_samples, prep.peaks, prep.native_rate,
                             prep.native_path, title=prep.title, source_id=prep.source_id)
            self._on_loaded(la)
            self._purge_cache(keep=self._top_dirs([prep.mix44_path, prep.native_path] +
                                                  [s.path for s in res.separation.stems]))
            self._on_separated(res.separation)
        except Exception:  # noqa: BLE001 - 홈 결과 파일은 이미 저장됨
            log.exception("could not load home result into advanced view")

    def _youtube_consent(self) -> bool:
        if self.settings.value("yt_consent", False, type=bool) or os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("YouTube 입력 안내")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("권리가 있는 콘텐츠만 사용해 주세요")
        box.setInformativeText(youtube.RIGHTS_NOTICE)
        ok = box.addButton("확인했습니다", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("취소", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is ok:
            self.settings.setValue("yt_consent", True)
            return True
        return False

    def open_youtube(self):
        """고급 옵션: YouTube 주소로 오디오를 받아 불러오기 (분리는 따로)."""
        if self.busy():
            return
        url, ok = QInputDialog.getText(self, "YouTube에서 불러오기", "YouTube 영상 주소:", QLineEdit.EchoMode.Normal, "")
        if not ok or not url.strip():
            return
        if not youtube.video_id(url):
            self._error("올바른 YouTube 영상 주소가 아닙니다.\n예: https://www.youtube.com/watch?v=…")
            return
        if not self._youtube_consent():
            return
        self._reset_tracks(clear_original=True)
        self._invalidate_key_cache()
        self._purge_cache()
        self.run_counter += 1
        self.yt_thread = YouTubeLoadThread(url, self.session_dir / f"yt{self.run_counter}",
                                           self.session_dir / f"yt{self.run_counter}" / "original.npy", self)
        (self.session_dir / f"yt{self.run_counter}").mkdir(parents=True, exist_ok=True)
        self.yt_thread.progress.connect(lambda f, m: (self._on_progress(f, m), self.set_status(m, "busy")))
        self.yt_thread.done.connect(lambda la, acq: (self._on_loaded(la), self.set_status(
            f"{la.name} — {acq.codec_name} {acq.abr_kbps:g}kbps {acq.sample_rate / 1000:g}kHz (재인코딩 없음). "
            "[AI 분리 시작]을 누르세요.")))
        self.yt_thread.failed.connect(lambda m: (self.set_status("YouTube 오디오를 가져오지 못했습니다.", "error"),
                                                 self.stack.setCurrentIndex(0) if self.loaded is None else None,
                                                 self._error(m)))
        self.yt_thread.finished.connect(lambda: (self._thread_finished("yt_thread"), self._work_finished()))
        self._show_progress(True, cancellable=True)
        self._on_progress(0.0, "YouTube 영상 정보 확인 중…")
        self.set_status("YouTube 오디오 가져오는 중…", "busy")
        self.yt_thread.start()
        self._set_state()

    def update_youtube(self):
        if self.yt_update_thread is not None:
            return
        t = YouTubeUpdateThread(self)
        self.yt_update_thread = t
        t.progress.connect(lambda f, m: self.set_status(m, "busy"))
        t.done.connect(self._on_yt_updated)
        t.failed.connect(lambda m: (self.set_status("업데이트 실패", "error"), self._error(m)))
        t.finished.connect(lambda: (setattr(self, "yt_update_thread", None), t.deleteLater()))
        self.set_status("YouTube 기능 업데이트 중…", "busy")
        t.start()

    def _on_yt_updated(self, ver: str):
        current = youtube.ytdlp_version()
        if youtube._version_tuple(ver) <= youtube._version_tuple(current):
            self.set_status(f"YouTube 기능이 이미 최신 버전입니다 (yt-dlp {current}).")
            if not os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
                QMessageBox.information(self, "YouTube 기능", f"이미 최신 버전입니다.\n(yt-dlp {current})")
            return
        self.set_status(f"YouTube 기능을 {ver} 로 업데이트했습니다. 다시 시작하면 적용됩니다.")
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return
        r = QMessageBox.question(self, "업데이트 완료",
                                 f"YouTube 다운로드 도구를 {ver} 버전으로 업데이트했습니다.\n"
                                 "프로그램을 다시 시작하면 적용됩니다. 지금 다시 시작할까요?")
        if r == QMessageBox.StandardButton.Yes and not self.busy():
            args = sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv
            QProcess.startDetached(sys.executable, args)
            self.close()

    # ======================================================= drag & drop ===
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and not self.busy():
            e.acceptProposedAction()
            if self.pages.currentIndex() == 1:
                self.drop_zone.set_hover(True)

    def dragLeaveEvent(self, e):
        self.drop_zone.set_hover(False)

    def dropEvent(self, e):
        self.drop_zone.set_hover(False)
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if not urls:
            return
        if self.pages.currentIndex() == 0:
            self.home_set_file(Path(urls[0].toLocalFile()))
        else:
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
            "AI 음원 분리: BS-RoFormer (코드 MIT · SW 모델 by jarredou, 커뮤니티 모델) · "
            "Demucs (Meta AI Research, MIT)<br>"
            f"키 조절: Rubber Band Library R3 (GPL, 별도 실행 파일 rbshift) · 현재 엔진: {pitch.engine_name() or '없음'}<br>"
            f"YouTube: yt-dlp {youtube.ytdlp_version()} (Unlicense) · Deno (MIT)<br>"
            "리샘플링: soxr (LGPL) · 오디오 변환: FFmpeg · MP3 인코더: LAME<br>"
            "GUI: Qt for Python / PySide6 (LGPLv3) · 글꼴: Pretendard (SIL OFL 1.1)<br>"
            f"PyTorch {tv} · Python {platform.python_version()} · {platform.system()} {platform.machine()}<br><br>"
            "분리된 음원의 저작권은 원곡 권리자에게 있습니다.")

    def closeEvent(self, e):
        if self.busy():
            r = QMessageBox.question(self, APP_NAME, "작업이 진행 중입니다. 취소하고 종료할까요?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.cancel_work()
            if self.pipeline_thread is not None:
                self.pipeline_thread.cancel()
            for t in (self.sep_thread, self.export_thread, self.load_thread, self.pipeline_thread, self.yt_thread):
                if t is not None and t.isRunning():
                    t.wait(20000)
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


HELP_TEXT = """[홈 화면 — 원클릭]
1. 음원 파일을 끌어다 놓거나 [파일 선택], 또는 YouTube 영상 주소를 붙여넣습니다.
2. [보컬만] 또는 [반주(MR)만]을 고릅니다 (기본: 반주).
3. 키 조절: −/+ 또는 슬라이더 (−12 ~ +12 반음, 템포·길이는 그대로). 0 이면 처리하지 않습니다.
4. [음원 추출 시작] → 진행 상황이 단계별로 표시되고, 끝나면 저장 폴더에 파일이 생깁니다.
   • 기본 출력: WAV 24-bit (설정에서 FLAC·MP3 320kbps 선택)
   • YouTube 는 권리가 있는 영상만 사용하세요.

[고급 옵션]
• 처리 방식: 속도 우선(Demucs) / 품질 우선(BS-RoFormer) / 최고 품질(겹침 4배)
  기본값: CPU 버전 = 속도 우선, CUDA 버전·Apple Silicon Mac = 품질 우선
• AI 모델·처리 장치 선택, 트랙별 재생(M 음소거 · S 솔로 · 볼륨), 파형 확대
• MR 저장(가이드 보컬 작게 남기기 가능) · 보컬만 저장 · 선택/전체 트랙 저장
• 키 조절과 '보컬 음색(포먼트) 유지' — 보컬 결과물에만 적용

• NVIDIA GPU 또는 Apple Silicon Mac에서는 GPU로 훨씬 빠르게 처리됩니다.
• GPU에서 문제가 생기면 자동으로 CPU로 다시 처리합니다.
• 처리 중 [취소]를 누르면 바로 멈추고 임시 파일을 정리합니다."""
