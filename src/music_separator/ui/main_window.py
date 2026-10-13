"""메인 윈도우."""
from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, audio_io, engine, models, paths, system
from ..dsp import audible_flags
from ..player import Mixer, Player, Track
from . import theme
from .widgets import HEADER_WIDTH, DropZone, SegmentedControl, TimeRuler, TrackRow
from .workers import ExportJob, ExportThread, LoadedAudio, LoadThread, SeparationThread

log = logging.getLogger(__name__)


class DeviceThread(QThread):
    done = Signal(list)

    def run(self):
        try:
            self.done.emit(system.detect_devices())
        except Exception as exc:  # torch 로드 실패 등
            log.exception("device detection failed")
            self.done.emit([system.DeviceInfo("cpu", f"CPU (장치 감지 실패: {exc})")])


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(9)
    t = QLabel(title.upper())
    t.setObjectName("SectionTitle")
    lay.addWidget(t)
    return card, lay


def _hint(text: str, name: str = "Hint") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(name)
    lbl.setWordWrap(True)
    return lbl


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[QSettings] = None):
        super().__init__()
        self.settings = settings or QSettings()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1060, 700)
        self.resize(1320, 840)
        self.setAcceptDrops(True)

        self.session_dir = paths.new_session_dir()
        self.run_counter = 0
        self.mixer = Mixer()
        self.player = Player(self.mixer)
        self.loaded: Optional[LoadedAudio] = None
        self.result: Optional[engine.SeparationResult] = None
        self.rows: dict[str, TrackRow] = {}
        self.chips: dict[str, QPushButton] = {}
        self.devices: list[system.DeviceInfo] = []
        self.load_thread: Optional[LoadThread] = None
        self.sep_thread: Optional[SeparationThread] = None
        self.export_thread: Optional[ExportThread] = None
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

        self.device_thread = DeviceThread(self)
        self.device_thread.done.connect(self._on_devices)
        self.device_thread.start()

    # ================================================================ UI ===
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_control_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_inspector())
        line = QFrame()
        line.setFixedWidth(1)
        line.setStyleSheet("background:#111;")
        body.addWidget(line)
        body.addWidget(self._build_main_area(), 1)
        rl.addLayout(body, 1)
        self.setCentralWidget(root)

        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.status_msg = QLabel("파일을 열어 시작하세요.")
        self.device_status = QLabel("장치 확인 중…")
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.device_status)
        sb.addPermanentWidget(QLabel(f"  v{__version__}  "))

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("ControlBar")
        bar.setFixedHeight(62)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(10)

        logo = QLabel("◉")
        logo.setStyleSheet(f"color:{theme.ACCENT}; font-size:20px;")
        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        self.file_title = QLabel("")
        self.file_title.setObjectName("FileTitle")
        self.file_title.setMaximumWidth(300)
        left = QHBoxLayout()
        left.setSpacing(8)
        left.addWidget(logo)
        left.addWidget(title)
        left.addSpacing(6)
        left.addWidget(self.file_title, 1)
        lw = QWidget()
        lw.setLayout(left)
        lay.addWidget(lw, 1)

        def tbtn(kind, tip, slot, w=40):
            b = QPushButton()
            b.setObjectName("Transport")
            b.setIcon(theme.make_icon(kind))
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(w, 34)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        self.rew_btn = tbtn("rewind", "처음으로 (Home)", self.rewind)
        self.play_btn = tbtn("play", "재생 / 일시정지 (Space)", self.toggle_play, 52)
        self.play_btn.setCheckable(True)
        self.stop_btn = tbtn("stop", "정지", self.stop)

        lcd = QFrame()
        lcd.setObjectName("Lcd")
        lcd.setFixedSize(230, 46)
        ll = QHBoxLayout(lcd)
        ll.setContentsMargins(14, 2, 14, 2)
        ll.setSpacing(10)
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        self.lcd_time = QLabel("00:00.0")
        self.lcd_time.setObjectName("LcdTime")
        self.lcd_time.setFont(theme.mono_font(19, True))
        sub = QLabel("POSITION")
        sub.setObjectName("LcdSub")
        tcol.addWidget(self.lcd_time)
        tcol.addWidget(sub)
        dcol = QVBoxLayout()
        dcol.setSpacing(0)
        self.lcd_total = QLabel("00:00")
        self.lcd_total.setObjectName("LcdTime")
        self.lcd_total.setFont(theme.mono_font(13))
        self.lcd_total.setStyleSheet(f"color:{theme.TEXT_DIM};")
        sub2 = QLabel("LENGTH")
        sub2.setObjectName("LcdSub")
        dcol.addWidget(self.lcd_total)
        dcol.addWidget(sub2)
        ll.addLayout(tcol)
        ll.addStretch(1)
        ll.addLayout(dcol)
        lay.addSpacing(6)
        lay.addWidget(lcd)
        lay.addSpacing(6)

        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setObjectName("Seek")
        self.seek.setRange(0, 10000)
        self.seek.setMinimumWidth(120)
        self.seek.setMaximumWidth(360)
        self.seek.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.seek.setToolTip("재생 위치")
        self.seek.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seek.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.seek.sliderReleased.connect(self._seek_released)
        self.seek.valueChanged.connect(self._seek_moved)
        lay.addWidget(self.seek, 1)
        lay.addSpacing(8)

        spk = QLabel()
        spk.setPixmap(theme.make_icon("speaker", theme.TEXT_DIM).pixmap(18, 18))
        lay.addWidget(spk)
        self.master = QSlider(Qt.Orientation.Horizontal)
        self.master.setRange(0, 100)
        self.master.setValue(80)
        self.master.setFixedWidth(110)
        self.master.setToolTip("마스터 볼륨 (재생에만 적용)")
        self.master.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.master.valueChanged.connect(lambda v: setattr(self.mixer, "master", v / 100.0))
        self.mixer.master = 0.8
        lay.addWidget(self.master)
        lay.addSpacing(10)
        self.open_btn = QPushButton("파일 열기")
        self.open_btn.setIcon(theme.make_icon("folder"))
        self.open_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.open_btn.clicked.connect(self.browse_file)
        lay.addWidget(self.open_btn)
        return bar

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        panel.setObjectName("Inspector")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 14, 12, 14)
        lay.setSpacing(12)

        # --- 분리 ---
        card, cl = _card("1  AI 음원 분리")
        self.mode_ctl = SegmentedControl([("2", "2-Stem"), ("4", "4-Stem"), ("6", "6-Stem")])
        self.mode_ctl.changed.connect(self._on_mode)
        cl.addWidget(self.mode_ctl)
        self.mode_desc = QLabel("")
        self.mode_desc.setStyleSheet("font-weight:600;")
        self.mode_desc.setWordWrap(True)
        cl.addWidget(self.mode_desc)
        self.mode_note = _hint("")
        cl.addWidget(self.mode_note)
        self.hq_box = QCheckBox("고품질 모드 (처리 시간 증가)")
        self.hq_box.toggled.connect(self._on_mode_options)
        cl.addWidget(self.hq_box)
        self.hq_hint = _hint("")
        cl.addWidget(self.hq_hint)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("처리 장치"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("자동 (권장)", "auto")
        self.device_combo.addItem("GPU", "gpu")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._on_device_pref)
        dev_row.addWidget(self.device_combo, 1)
        cl.addLayout(dev_row)
        self.device_hint = _hint("장치 확인 중…")
        cl.addWidget(self.device_hint)

        self.start_btn = QPushButton("분리 시작")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_btn.clicked.connect(self.start_separation)
        cl.addWidget(self.start_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedHeight(10)
        self.progress.hide()
        cl.addWidget(self.progress)
        prow = QHBoxLayout()
        self.progress_lbl = _hint("")
        self.progress_lbl.hide()
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cancel_btn.clicked.connect(self.cancel_work)
        self.cancel_btn.hide()
        prow.addWidget(self.progress_lbl, 1)
        prow.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignTop)
        cl.addLayout(prow)
        self.notice_lbl = _hint("", "Notice")
        self.notice_lbl.hide()
        cl.addWidget(self.notice_lbl)
        lay.addWidget(card)

        # --- MR ---
        card, cl = _card("2  MR(반주) 만들기")
        cl.addWidget(_hint("MR에서 뺄 악기를 누르세요. 나머지 트랙을 합쳐 새 반주를 만듭니다."))
        self.chip_box = QWidget()
        self.chip_grid = QGridLayout(self.chip_box)
        self.chip_grid.setContentsMargins(0, 2, 0, 2)
        self.chip_grid.setHorizontalSpacing(6)
        self.chip_grid.setVerticalSpacing(6)
        cl.addWidget(self.chip_box)
        self.chip_empty = _hint("분리를 완료하면 악기 목록이 표시됩니다.")
        cl.addWidget(self.chip_empty)
        self.mr_volume_box = QCheckBox("트랙 볼륨 설정 반영")
        self.mr_volume_box.setToolTip("체크하면 각 트랙의 볼륨 슬라이더 값이 MR에 적용됩니다.")
        cl.addWidget(self.mr_volume_box)
        mr_row = QHBoxLayout()
        self.mr_preview_btn = QPushButton("미리듣기")
        self.mr_preview_btn.setToolTip("선택한 악기를 음소거하고 재생합니다.")
        self.mr_preview_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mr_preview_btn.clicked.connect(self.preview_mr)
        self.mr_save_btn = QPushButton("MR 저장")
        self.mr_save_btn.setObjectName("Primary")
        self.mr_save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mr_save_btn.clicked.connect(self.export_mr)
        mr_row.addWidget(self.mr_preview_btn)
        mr_row.addWidget(self.mr_save_btn, 1)
        cl.addLayout(mr_row)
        lay.addWidget(card)

        lay.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_export_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("ExportBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 9, 14, 9)
        lay.setSpacing(8)
        t = QLabel("3  저장")
        t.setObjectName("SectionTitle")
        lay.addWidget(t)
        lay.addSpacing(6)
        lay.addWidget(QLabel("폴더"))
        self.folder_lbl = QLabel("")
        self.folder_lbl.setObjectName("Hint")
        self.folder_lbl.setMinimumWidth(80)
        self.folder_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.folder_lbl, 1)
        change = QPushButton("변경…")
        change.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        change.setToolTip("저장 폴더 선택 (곡마다 하위 폴더가 만들어집니다)")
        change.clicked.connect(self.choose_output_dir)
        opn = QPushButton()
        opn.setIcon(theme.make_icon("folder"))
        opn.setToolTip("저장 폴더 열기")
        opn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        opn.clicked.connect(self.open_output_dir)
        lay.addWidget(change)
        lay.addWidget(opn)
        lay.addSpacing(10)
        self.fmt_combo = QComboBox()
        for key, (label, _) in audio_io.EXPORT_FORMATS.items():
            self.fmt_combo.addItem(label, key)
        self.fmt_combo.setToolTip("저장 파일 형식")
        lay.addWidget(self.fmt_combo)
        self.export_sel_btn = QPushButton("선택한 트랙 저장")
        self.export_sel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_sel_btn.setToolTip("트랙 이름 옆 '저장'에 체크된 트랙을 각각 파일로 저장합니다.\n"
                                       "개별 트랙은 볼륨 슬라이더와 관계없이 원래 음량으로 저장됩니다.")
        self.export_sel_btn.clicked.connect(self.export_selected)
        self.export_all_btn = QPushButton("전체 트랙 저장")
        self.export_all_btn.setObjectName("Primary")
        self.export_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_all_btn.clicked.connect(self.export_all)
        lay.addWidget(self.export_sel_btn)
        lay.addWidget(self.export_all_btn)
        return bar

    def _build_main_area(self) -> QWidget:
        self.stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.drop_zone.browseRequested.connect(self.browse_file)
        self.stack.addWidget(self.drop_zone)

        tracks_page = QWidget()
        tracks_page.setObjectName("TrackArea")
        tl = QVBoxLayout(tracks_page)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        ruler_row = QHBoxLayout()
        ruler_row.setContentsMargins(0, 0, 0, 0)
        ruler_row.setSpacing(0)
        corner = QLabel("  트랙")
        corner.setFixedSize(HEADER_WIDTH, 26)
        corner.setStyleSheet(f"background:#26262a; color:{theme.TEXT_FAINT}; font-size:11px; "
                             "border-right:1px solid #111; border-bottom:1px solid #111;")
        self.ruler = TimeRuler()
        self.ruler.seekRequested.connect(self.seek_fraction)
        rw = QWidget()
        rwl = QHBoxLayout(rw)
        rwl.setContentsMargins(4, 0, 6, 0)
        rwl.addWidget(self.ruler)
        ruler_row.addWidget(corner)
        ruler_row.addWidget(rw, 1)
        tl.addLayout(ruler_row)

        self.track_scroll = QScrollArea()
        self.track_scroll.setWidgetResizable(True)
        self.track_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # 주의: 선택자 없는 스타일시트는 모든 자식 위젯(M/S 버튼 등)에 상속되므로 반드시 선택자 지정
        self.track_scroll.setStyleSheet(f"QScrollArea {{ background:{theme.TRACK_BG}; border:none; }}")
        self.track_scroll.viewport().setObjectName("TrackArea")
        self.track_container = QWidget()
        self.track_container.setObjectName("TrackArea")
        self.track_layout = QVBoxLayout(self.track_container)
        self.track_layout.setContentsMargins(0, 0, 0, 0)
        self.track_layout.setSpacing(0)
        self.track_layout.addStretch(1)
        self.track_scroll.setWidget(self.track_container)
        tl.addWidget(self.track_scroll, 1)
        tl.addWidget(self._build_export_bar())
        self.empty_hint = QLabel("왼쪽의 [분리 시작]을 누르면 분리된 트랙이 여기에 표시됩니다.")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setStyleSheet(f"color:{theme.TEXT_FAINT}; padding:24px; font-size:14px;")
        self.track_layout.insertWidget(self.track_layout.count() - 1, self.empty_hint)
        self.stack.addWidget(tracks_page)
        return self.stack

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("파일")
        a = QAction("파일 열기…", self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self.browse_file)
        fm.addAction(a)
        a = QAction("저장 폴더 변경…", self)
        a.triggered.connect(self.choose_output_dir)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("종료", self)
        a.setShortcut(QKeySequence.StandardKey.Quit)
        a.setMenuRole(QAction.MenuRole.QuitRole)
        a.triggered.connect(self.close)
        fm.addAction(a)

        pm = mb.addMenu("재생")
        a = QAction("재생 / 일시정지  (Space)", self)
        a.triggered.connect(self.toggle_play)
        pm.addAction(a)
        a = QAction("처음으로  (Home)", self)
        a.triggered.connect(self.rewind)
        pm.addAction(a)
        a = QAction("모든 음소거·솔로 해제", self)
        a.triggered.connect(self.reset_mixer)
        pm.addAction(a)

        hm = mb.addMenu("도움말")
        a = QAction("사용 방법", self)
        a.triggered.connect(self.show_help)
        hm.addAction(a)
        hm.addSeparator()
        a = QAction("AI 모델 폴더 열기", self)
        a.triggered.connect(lambda: paths.open_in_file_manager(paths.user_models_dir()))
        hm.addAction(a)
        a = QAction("AI 모델 가져오기 (오프라인 설치)…", self)
        a.triggered.connect(self.import_models)
        hm.addAction(a)
        a = QAction("로그 폴더 열기", self)
        a.triggered.connect(lambda: paths.open_in_file_manager(paths.logs_dir()))
        hm.addAction(a)
        hm.addSeparator()
        a = QAction(f"{APP_NAME} 정보", self)
        a.setMenuRole(QAction.MenuRole.AboutRole)
        a.triggered.connect(self.show_about)
        hm.addAction(a)

    # ========================================================= settings ===
    def _restore_settings(self):
        s = self.settings
        self.output_dir = Path(s.value("out_dir", str(paths.default_output_dir())))
        self._update_folder_label()
        self.mode_ctl.set_current(str(s.value("mode", "4")))
        self.hq_box.setChecked(s.value("hq", False, type=bool))
        idx = self.device_combo.findData(s.value("device", "auto"))
        self.device_combo.setCurrentIndex(max(0, idx))
        idx = self.fmt_combo.findData(s.value("fmt", "wav16"))
        self.fmt_combo.setCurrentIndex(max(0, idx))
        self.mr_volume_box.setChecked(s.value("mr_volume", True, type=bool))
        self._on_mode(self.mode_ctl.current(), save=False)

    def _save_settings(self):
        s = self.settings
        s.setValue("out_dir", str(self.output_dir))
        s.setValue("mode", self.mode_ctl.current())
        s.setValue("hq", self.hq_box.isChecked())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("fmt", self.fmt_combo.currentData())
        s.setValue("mr_volume", self.mr_volume_box.isChecked())
        s.sync()

    def _update_folder_label(self):
        text = str(self.output_dir)
        home = str(Path.home())
        if text.startswith(home):
            text = "~" + text[len(home):]
        self.folder_lbl.setText(text)
        self.folder_lbl.setToolTip(str(self.output_dir))

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
        for w in (self.mr_preview_btn, self.mr_save_btn, self.export_sel_btn, self.export_all_btn):
            w.setEnabled(has_result and not busy)
        for w in (self.play_btn, self.stop_btn, self.rew_btn, self.seek):
            w.setEnabled(has_file)
        self.chip_empty.setVisible(not has_result)
        self.start_btn.setText("다시 분리" if has_result else "분리 시작")

    def _on_mode(self, key: str, save: bool = True):
        mode = engine.MODES[key]
        self.mode_ctl.set_current(key)
        self.mode_desc.setText(mode.description)
        self.mode_note.setText(mode.note)
        self.mode_note.setObjectName("Note" if key == "6" else "Hint")
        self.mode_note.style().unpolish(self.mode_note)
        self.mode_note.style().polish(self.mode_note)
        self._on_mode_options()
        if save:
            self._save_settings()

    def _on_mode_options(self, *_):
        key = self.mode_ctl.current()
        hq = self.hq_box.isChecked()
        model = engine.model_for(key, hq)
        spec = models.MODELS[model]
        if key == "6":
            tip = "고품질: 2회 처리 후 평균 (처리 시간 약 2배, 품질 소폭 향상)"
        else:
            tip = "고품질: 미세조정 모델 4개(htdemucs_ft) 사용 (처리 시간 약 4배, 품질 향상)"
        self.hq_box.setToolTip(tip)
        status = "설치됨" if models.is_installed(model) else f"첫 사용 시 약 {spec.approx_mb}MB 다운로드"
        self.hq_hint.setText(f"AI 모델: {model} · {status}")
        self.hq_hint.setToolTip(tip)
        self.settings.setValue("hq", hq)

    def _on_device_pref(self, *_):
        self._update_device_hint()
        self.settings.setValue("device", self.device_combo.currentData())

    def _on_devices(self, devices: list):
        self.devices = devices
        gpus = [d for d in devices if d.key != "cpu"]
        model = self.device_combo.model()
        item = model.item(1)
        if item is not None:
            item.setEnabled(bool(gpus))
            if gpus:
                self.device_combo.setItemText(1, f"GPU — {gpus[0].label}")
            else:
                self.device_combo.setItemText(1, "GPU (사용 불가)")
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
            dev = cpu
            msg = f"{cpu.label}" + (" · GPU 없음 (CPU는 곡 길이만큼 걸릴 수 있음)" if not gpus else "")
        else:
            dev = gpus[0]
            msg = f"GPU 가속: {dev.label} · 문제 시 CPU로 자동 전환"
        self.device_hint.setText(msg)
        self.device_status.setText(f"처리 장치: {dev.label}  ")

    def _resolve_device(self) -> str:
        pref = self.device_combo.currentData()
        if pref == "cpu":
            return "cpu"
        gpus = [d for d in self.devices if d.key != "cpu"] if self.devices else []
        if not self.devices:
            return system.choose_device(pref).key
        return gpus[0].key if gpus else "cpu"

    def set_status(self, text: str):
        self.status_msg.setText(text)

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
        need = 0
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
        self._purge_cache()
        self.set_status(f"불러오는 중: {path.name}")
        self.file_title.setText("— 불러오는 중…")
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
        self.file_title.setText(fm.elidedText(f"— {la.source.name}", Qt.TextElideMode.ElideMiddle, 290))
        self.file_title.setToolTip(str(la.source))
        self.lcd_total.setText(audio_io.format_time(la.seconds))
        self.ruler.set_duration(la.seconds)
        self.stack.setCurrentIndex(1)
        orig = self._np_load(la.mix_path)
        self._add_row("original", "원본", "Original · 참고용", engine.stem_color("original"), la.peaks, True)
        self.mixer.set_tracks([Track("original", orig, reference=True)])
        self.mixer.seek(0)
        self.empty_hint.show()
        self.set_status(f"불러옴: {la.source.name} ({audio_io.format_time(la.seconds)}) — "
                        "모드를 고르고 [분리 시작]을 누르세요.")
        self._set_state()

    def _on_load_failed(self, msg: str):
        self.file_title.setText("")
        self.set_status("파일을 불러오지 못했습니다.")
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
        model_name = os.environ.get("MUSIC_SEPARATOR_FORCE_MODEL") or engine.model_for(mode, hq)
        interactive = not os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE")

        if model_name != "demucs_unittest" and not models.is_installed(model_name):
            spec = models.MODELS[model_name]
            free = system.free_disk(paths.user_models_dir())
            if free < spec.approx_mb * system.MB * 1.2:
                self._error(f"AI 모델을 저장할 공간이 부족합니다.\n필요: 약 {spec.approx_mb}MB, "
                            f"남은 공간: {system.human_bytes(free)}")
                return
            if interactive:
                r = QMessageBox.question(
                    self, "AI 모델 다운로드",
                    f"이 모드는 AI 모델 '{model_name}'(약 {spec.approx_mb}MB)이 필요합니다.\n"
                    "처음 한 번만 인터넷에서 다운로드하며, 이후에는 인터넷 없이 사용할 수 있습니다.\n\n"
                    "지금 다운로드할까요?")
                if r != QMessageBox.StandardButton.Yes:
                    return
        if mode == "6" and interactive and not self.settings.value("warned_6s", False, type=bool):
            QMessageBox.information(
                self, "6-Stem 안내",
                "6-Stem 모델(htdemucs_6s)은 Demucs 공식 '실험적' 모델입니다.\n\n"
                "• 기타: 비교적 잘 분리되지만 다른 악기 소리가 섞일 수 있습니다.\n"
                "• 피아노: 품질이 낮은 편입니다. 소리가 작게 나오거나 '나머지 악기'로 새는 경우가 많습니다.\n"
                "• 보컬·드럼·베이스: 4-Stem과 비슷한 수준입니다.\n\n"
                "깨끗한 MR이 목적이라면 2-Stem 또는 4-Stem을 권장합니다.")
            self.settings.setValue("warned_6s", True)

        m = engine.MODES[mode]
        n_src = 6 if mode == "6" else 4
        bag = len(models.MODELS[model_name].sigs) if model_name in models.MODELS else 1
        req = system.separation_requirement(self.loaded.n_samples, n_src, len(m.outputs),
                                            self.session_dir, bag)
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
        self.sep_thread = SeparationThread(self.loaded.mix_path, mode, hq, device, out_dir, model_name, self)
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
        self.set_status(f"{m.title} 분리 중… 처리 중에도 다른 창을 사용할 수 있습니다.")

    def _show_progress(self, on: bool):
        self.progress.setVisible(on)
        self.progress_lbl.setVisible(on)
        self.cancel_btn.setVisible(on)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("취소")

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
        self.set_status("분리 실패")
        self._error(msg)

    def _on_separated(self, res: engine.SeparationResult):
        self._reset_tracks(clear_original=False)
        run_dir = res.stems[0].path.parent.name if res.stems else ""
        self._purge_cache(keep={self.loaded.mix_path.name, run_dir})
        self.result = res
        tracks = [t for t in self.mixer.tracks if t.key == "original"]
        if tracks:
            tracks[0].mute = True
        for stem in res.stems:
            label = engine.stem_label(stem.key, res.mode)
            self._add_row(stem.key, label, engine.stem_english(stem.key), engine.stem_color(stem.key),
                          stem.peaks)
            tracks.append(Track(stem.key, self._np_load(stem.path)))
        self.mixer.set_tracks(tracks)
        if "original" in self.rows:
            self.rows["original"].set_muted(True)
        self.empty_hint.hide()
        self._build_chips(res)
        self._refresh_audible()
        dev = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(res.device, res.device)
        self.set_status(f"분리 완료 — {engine.MODES[res.mode].title}, {dev}, "
                        f"{audio_io.format_time(res.elapsed)} 소요. ▶ 재생해서 들어보세요.")
        self._set_state()

    # ============================================================ tracks ===
    def _add_row(self, key, name, sub, color, peaks, reference=False):
        row = TrackRow(key, name, sub, color, peaks, reference)
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
            self.lcd_total.setText("00:00")
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

    # ============================================================== MR ===
    def _build_chips(self, res: Optional[engine.SeparationResult]):
        for b in self.chips.values():
            b.setParent(None)
            b.deleteLater()
        self.chips.clear()
        if res is None:
            return
        for i, stem in enumerate(res.stems):
            b = QPushButton(engine.stem_label(stem.key, res.mode))
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip("누르면 MR에서 제외됩니다")
            b.setChecked(stem.key == "vocals")
            self.chip_grid.addWidget(b, i // 3, i % 3)
            self.chips[stem.key] = b

    def _mr_selection(self) -> tuple[list[str], list[str]]:
        excluded = [k for k, b in self.chips.items() if b.isChecked()]
        included = [k for k in self.chips if k not in excluded]
        return included, excluded

    def preview_mr(self):
        included, excluded = self._mr_selection()
        for key, row in self.rows.items():
            row.set_solo(False)
            row.set_muted(key == "original" or key in excluded)
        self._refresh_audible()
        if not self.player.playing:
            self.toggle_play()
        self.set_status("MR 미리듣기: " + (", ".join(self._label(k) for k in excluded) or "없음") + " 제외")

    def export_mr(self):
        if self.result is None:
            return
        included, excluded = self._mr_selection()
        if not included:
            self._error("MR에 포함할 트랙이 없습니다. 제외할 악기를 줄여 주세요.")
            return
        use_vol = self.mr_volume_box.isChecked()
        sources = [(self._track(k).data, self.rows[k].gain if use_vol else 1.0) for k in included]
        fmt = self.fmt_combo.currentData()
        ext = audio_io.EXPORT_FORMATS[fmt][1]
        if excluded:
            suffix = "MR (" + ", ".join(self._label(k) for k in excluded) + " 제외)"
        else:
            suffix = "전체 믹스"
        path = audio_io.unique_path(self._song_dir() / audio_io.safe_filename(f"{self._song()} - {suffix}{ext}"))
        self._run_export([ExportJob(path, sources, fmt, suffix)])

    # ========================================================== export ===
    def _song(self) -> str:
        return audio_io.safe_filename(self.loaded.source.stem if self.loaded else "track", 80)

    def _song_dir(self) -> Path:
        return self.output_dir / self._song()

    def _label(self, key: str) -> str:
        return engine.stem_label(key, self.result.mode if self.result else "4")

    def _stem_jobs(self, keys: list[str]) -> list[ExportJob]:
        fmt = self.fmt_combo.currentData()
        ext = audio_io.EXPORT_FORMATS[fmt][1]
        jobs = []
        for k in keys:
            name = f"{self._song()} - {self._label(k)} ({engine.stem_english(k)}){ext}"
            path = audio_io.unique_path(self._song_dir() / audio_io.safe_filename(name))
            jobs.append(ExportJob(path, [(self._track(k).data, 1.0)], fmt, self._label(k)))
        return jobs

    def export_selected(self):
        if self.result is None:
            return
        keys = [s.key for s in self.result.stems if self.rows[s.key].selected]
        if not keys:
            self._error("저장할 트랙을 선택하세요. (트랙 이름 옆 '저장' 체크)")
            return
        self._run_export(self._stem_jobs(keys))

    def export_all(self):
        if self.result is None:
            return
        self._run_export(self._stem_jobs([s.key for s in self.result.stems]))

    def _run_export(self, jobs: list[ExportJob]):
        if self.busy() or not jobs:
            return
        n = self.mixer.length
        need = sum(audio_io.estimate_export_bytes(n, j.fmt) for j in jobs) + 20 * system.MB
        free = system.free_disk(self._song_dir())
        if free < need:
            self._error(f"저장 폴더의 디스크 공간이 부족합니다.\n필요: {system.human_bytes(need)}, "
                        f"남은 공간: {system.human_bytes(free)}")
            return
        try:
            self._song_dir().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._error(f"저장 폴더를 만들 수 없습니다: {self._song_dir()}\n({exc})\n"
                        "[변경]을 눌러 다른 폴더를 선택해 주세요.")
            return
        self.export_thread = ExportThread(jobs, self)
        self.export_thread.progress.connect(self._on_progress)
        self.export_thread.done.connect(self._on_exported)
        self.export_thread.failed.connect(lambda m: (self.set_status("저장 실패"), self._error(m)))
        self.export_thread.finished.connect(
            lambda: (self._thread_finished("export_thread"), self._work_finished()))
        self._show_progress(True)
        self.cancel_btn.hide()
        self.export_thread.start()
        self._set_state()

    def _on_exported(self, files: list):
        self.set_status(f"{len(files)}개 파일 저장 완료 → {self._song_dir()}")
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            return
        box = QMessageBox(self)
        box.setWindowTitle("저장 완료")
        box.setText(f"{len(files)}개 파일을 저장했습니다.")
        box.setInformativeText("\n".join(Path(f).name for f in files[:8]) + ("\n…" if len(files) > 8 else ""))
        open_btn = box.addButton("폴더 열기", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("닫기", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            paths.open_in_file_manager(self._song_dir())

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", str(self.output_dir))
        if d:
            self.output_dir = Path(d)
            self._update_folder_label()
            self._save_settings()

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
        self._sync_play_btn()

    def stop(self):
        self.player.stop()
        self._sync_play_btn()

    def rewind(self):
        self.mixer.seek(0)

    def seek_fraction(self, frac: float):
        self.mixer.seek(int(frac * self.mixer.length))

    def _seek_moved(self, v: int):
        if self._seeking:
            self.seek_fraction(v / 10000)

    def _seek_released(self):
        self._seeking = False
        self.seek_fraction(self.seek.value() / 10000)

    def _sync_play_btn(self):
        playing = self.player.playing
        self.play_btn.setChecked(playing)
        self.play_btn.setIcon(theme.make_icon("pause" if playing else "play"))

    def _tick(self):
        if self.play_btn.isChecked() != self.player.playing:
            self._sync_play_btn()
        if self.mixer.length == 0:
            return
        sec = self.player.display_seconds()
        frac = min(1.0, sec / max(1e-6, self.mixer.duration))
        m, s = divmod(sec, 60)
        self.lcd_time.setText(f"{int(m):02d}:{s:04.1f}")
        if not self._seeking:
            self.seek.blockSignals(True)
            self.seek.setValue(int(frac * 10000))
            self.seek.blockSignals(False)
        self.ruler.set_position(frac)
        for row in self.rows.values():
            row.wave.set_position(frac)

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
        if os.environ.get("MUSIC_SEPARATOR_NONINTERACTIVE"):
            self.last_error = msg
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
            "GUI: Qt for Python / PySide6 (LGPLv3)<br>"
            "오디오 변환: FFmpeg · MP3 인코더: LAME<br>"
            f"PyTorch {tv} · Python {platform.python_version()} · {platform.system()} {platform.machine()}<br><br>"
            "분리된 음원의 저작권은 원곡 권리자에게 있습니다. 개인 연습·학습 용도로 사용해 주세요.")

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
2. 왼쪽에서 분리 모드를 고릅니다.
   • 2-Stem: 보컬 / 반주
   • 4-Stem: 보컬 / 드럼 / 베이스 / 기타 악기
   • 6-Stem: 보컬 / 드럼 / 베이스 / 기타 / 피아노 / 나머지 (실험적)
3. [분리 시작]을 누릅니다. 처음 한 번은 AI 모델을 다운로드합니다.
4. 분리가 끝나면 ▶(Space)로 재생합니다.
   • M: 음소거 · S: 솔로(이 트랙만 듣기) · 슬라이더: 볼륨
   • 파형이나 위쪽 시간 눈금을 클릭하면 그 위치로 이동합니다.
5. MR 만들기: 뺄 악기를 누르고 [MR 저장].
   (예: 보컬만 빼면 노래방 MR, 드럼만 빼면 드럼 연습용 MR)
6. 저장: 형식(WAV/MP3)과 폴더를 정하고 [선택한 트랙 저장] 또는 [전체 트랙 저장].

팁
• NVIDIA GPU 또는 Apple Silicon Mac에서는 GPU로 훨씬 빠르게 처리됩니다.
• GPU에서 문제가 생기면 자동으로 CPU로 다시 처리합니다.
• 처리 중 [취소]를 누르면 바로 멈춥니다."""
