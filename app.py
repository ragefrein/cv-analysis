import os
import sys
import traceback
import base64

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QPointF, QRectF, QSize
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget, QComboBox, QDialog, QDialogButtonBox,
    QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QScrollArea,
)

from vision import analyze_video, draw_overlay
from players import TEAM_COLORS, TEAM_NAMES, merge_players, player_label, player_team
from storage import load_session, save_session, session_path
from tactics import draw_tactical, set_calibration
from court import correct_boundary
from identity_review import refresh_review, split_player, verify_player, visible_players


def frame_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


class VideoView(QWidget):
    def __init__(self, minimum=(640, 360)):
        super().__init__()
        self._image = None
        self.setMinimumSize(*minimum)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, image: QImage):
        self._image = image
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._image is None:
            p.setPen(Qt.gray)
            p.drawText(self.rect(), Qt.AlignCenter, "Buka video futsal untuk mulai")
            return
        scaled = self._image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, QPixmap.fromImage(scaled))


class CalibrationView(VideoView):
    changed = Signal(int)

    def __init__(self, image, max_points=4):
        super().__init__((600, 360))
        self.set_image(image)
        self.points = []
        self.max_points = max_points

    def image_rect(self):
        scaled = self._image.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2,
                      scaled.width(), scaled.height())

    def mousePressEvent(self, event):
        rect = self.image_rect()
        if event.button() == Qt.RightButton and self.points:
            self.points.pop()
        elif event.button() == Qt.LeftButton and rect.contains(event.position()) and len(self.points) < self.max_points:
            point = event.position() - rect.topLeft()
            self.points.append([point.x() * self._image.width() / rect.width(),
                                point.y() * self._image.height() / rect.height()])
        self.changed.emit(len(self.points))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#38e8bb"), 2))
        rect = self.image_rect()
        points = [QPointF(rect.x() + x * rect.width() / self._image.width(),
                         rect.y() + y * rect.height() / self._image.height()) for x, y in self.points]
        for i, point in enumerate(points):
            painter.drawEllipse(point, 6, 6)
            painter.drawText(point + QPointF(10, -8), str(i + 1))
            if i:
                painter.drawLine(points[i - 1], point)
        if len(points) == self.max_points or (self.max_points > 4 and len(points) >= 3):
            painter.drawLine(points[-1], points[0])


class CalibrationDialog(QDialog):
    def __init__(self, image, parent):
        super().__init__(parent)
        self.setWindowTitle("Kalibrasi papan taktik")
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        hint = QLabel("Klik 4 sudut lapangan penuh secara berurutan: 1 kiri atas → 2 kanan atas → "
                      "3 kanan bawah → 4 kiri bawah. Klik kanan untuk membatalkan titik. "
                      "Gunakan frame yang memperlihatkan keempat sudut.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.canvas = CalibrationView(image)
        layout.addWidget(self.canvas, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.canvas.changed.connect(lambda n: buttons.button(QDialogButtonBox.Save).setEnabled(n == 4))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class BoundaryDialog(QDialog):
    def __init__(self, image, parent):
        super().__init__(parent)
        self.setWindowTitle("Batas area bermain yang terlihat")
        self.resize(1100, 760)
        layout = QVBoxLayout(self)
        hint = QLabel("Klik 3–12 titik mengikuti batas area bermain yang terlihat, berurutan mengelilingi lapangan. "
                      "Jangan masukkan perimeter merah, bangku, atau tribun. Klik kanan menghapus titik. "
                      "Koreksi mengikuti kamera dalam shot ini. Jalankan analisis ulang untuk memulihkan deteksi yang sebelumnya terlewat.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.canvas = CalibrationView(image, max_points=12)
        layout.addWidget(self.canvas, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.canvas.changed.connect(lambda n: buttons.button(QDialogButtonBox.Save).setEnabled(n >= 3))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class IdentityReviewDialog(QDialog):
    def __init__(self, tracks, parent):
        super().__init__(parent)
        self.tracks = tracks
        self.modified = False
        self.jump = None
        self.setWindowTitle("Tinjau fragmen identitas")
        self.resize(850, 620)
        layout = QVBoxLayout(self)
        hint = QLabel("Kandidat berikut hanya dugaan. Bandingkan tim, nomor punggung, wajah, dan waktu kemunculan. "
                      "Jarak penampilan lebih kecil berarti lebih mirip; angka ini bukan persentase akurasi.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select_link)
        layout.addWidget(self.list)
        pair = QHBoxLayout()
        self.source = QLabel()
        self.target = QLabel()
        for label in (self.source, self.target):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(160, 160)
            pair.addWidget(label)
        layout.addLayout(pair)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.same = QPushButton("Orang sama · gabungkan")
        self.same.clicked.connect(self.accept_link)
        self.different = QPushButton("Orang berbeda")
        self.different.clicked.connect(self.reject_link)
        self.inspect = QPushButton("Buka frame sumber")
        self.inspect.clicked.connect(lambda: self.inspect_link("source"))
        self.inspect_target = QPushButton("Buka frame tujuan")
        self.inspect_target.clicked.connect(lambda: self.inspect_link("target"))
        for button in (self.same, self.different, self.inspect, self.inspect_target):
            actions.addWidget(button)
        layout.addLayout(actions)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.refresh()

    def refresh(self):
        self.links = refresh_review(self.tracks)
        self.list.clear()
        for link in self.links:
            self.list.addItem(f"P{link['source']} → P{link['target']}  ·  jarak {link['score']:.3f}  ·  {link['reason']}")
        enabled = bool(self.links)
        for button in (self.same, self.different, self.inspect, self.inspect_target):
            button.setEnabled(enabled)
        if enabled:
            self.list.setCurrentRow(0)
        else:
            self.details.setText("Tidak ada kandidat gabung lain. Verifikasi tim/nomor pada daftar pemain; tanpa kandidat belum berarti semua ID benar.")
            self.source.clear()
            self.target.clear()

    def select_link(self, row):
        if row < 0 or row >= len(self.links):
            return
        link = self.links[row]
        for label, pid in ((self.source, link["source"]), (self.target, link["target"])):
            player = self.tracks["players"][str(pid)]
            pixmap = QPixmap()
            if player.get("portrait"):
                pixmap.loadFromData(base64.b64decode(player["portrait"]))
            if pixmap.isNull():
                label.setText(f"P{pid}")
            else:
                label.setPixmap(pixmap.scaled(110, 185, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        fps = self.tracks.get("fps", 30)
        source, target = (self.tracks["players"][str(link[key])] for key in ("source", "target"))
        self.details.setText(f"Sumber: {player_label(self.tracks, link['source'])} "
                             f"({source['first_seen']/fps:.1f}–{source['last_seen']/fps:.1f} dtk). "
                             f"Tujuan: {player_label(self.tracks, link['target'])} "
                             f"({target['first_seen']/fps:.1f}–{target['last_seen']/fps:.1f} dtk).")

    def accept_link(self):
        row = self.list.currentRow()
        if row < 0:
            return
        link = self.links[row]
        try:
            merge_players(self.tracks, link["source"], link["target"])
        except ValueError as exc:
            QMessageBox.information(self, "Periksa identitas", str(exc))
            return
        self.tracks.setdefault("review_decisions", []).append({"action": "same", "ids": [link["source"], link["target"]]})
        self.modified = True
        self.refresh()

    def reject_link(self):
        row = self.list.currentRow()
        if row < 0:
            return
        link = self.links[row]
        self.tracks.setdefault("review_decisions", []).append({"action": "different", "ids": [link["source"], link["target"]]})
        self.modified = True
        self.refresh()

    def inspect_link(self, which="source"):
        row = self.list.currentRow()
        if row >= 0:
            pid = self.links[row][which]
            self.jump = (pid, self.tracks["players"][str(pid)]["first_seen"])
            self.accept()


class AnalyzeThread(QThread):
    progress = Signal(int, str)

    def __init__(self, path, max_frames=None, initial_roster=None, floor_boundaries=None):
        super().__init__()
        self.path = path
        self.max_frames = max_frames
        self.initial_roster = initial_roster
        self.floor_boundaries = floor_boundaries
        self.result = None
        self.error = None
        self.save_error = None

    def run(self):
        try:
            self.result = analyze_video(
                self.path,
                lambda i, total, msg: self.progress.emit(
                    int(min(100, i * 100 / max(1, total))), msg
                ),
                max_frames=self.max_frames,
                cancelled=self.isInterruptionRequested,
                initial_roster=self.initial_roster,
                floor_boundaries=self.floor_boundaries,
            )
            try:
                self.progress.emit(99, "Menyimpan pemain dan hasil analisis…")
                save_session(self.path, self.result)
            except Exception as exc:
                self.save_error = str(exc)
        except Exception:
            self.error = traceback.format_exc()


class Card(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("Card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 16)
        self.layout.setSpacing(10)
        label = QLabel(title.upper())
        label.setObjectName("CardTitle")
        self.layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Futsal Vision")
        self.resize(1440, 900)

        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.current_idx = 0
        self.last_read = -2
        self.tracks = None
        self.previous_roster = None
        self.worker = None
        self.playing = False
        self.roster_items = {}
        self._closing = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_style()

        self.view.setFocusPolicy(Qt.StrongFocus)
        shortcut = QShortcut(QKeySequence(Qt.Key_Space), self.view, self.toggle_play)
        shortcut.setContext(Qt.WidgetShortcut)

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Futsal Vision")
        title.setObjectName("Title")
        subtitle = QLabel("Dua tim · skeleton tubuh · identitas pemain · papan taktik 2D")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        head.addLayout(title_box)
        head.addStretch()
        badge = QLabel("YOLO11 POSE  •  PLAYER MEMORY")
        badge.setObjectName("Badge")
        head.addWidget(badge, 0, Qt.AlignVCenter)
        outer.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        video_card = QFrame()
        video_card.setObjectName("VideoCard")
        v = QVBoxLayout(video_card)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)
        self.view = VideoView()
        v.addWidget(self.view, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.btn_open = QPushButton("Buka Video")
        self.btn_open.clicked.connect(self.open_video)
        self.btn_analyze = QPushButton("Analisis")
        self.btn_analyze.setObjectName("Primary")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_play = QPushButton("Putar")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("Time")
        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_analyze)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)
        v.addLayout(controls)

        tactical_row = QHBoxLayout()
        self.tactical_view = VideoView((300, 165))
        self.tactical_view.setMaximumHeight(235)
        tactical_row.addWidget(self.tactical_view, 2)
        tactical_info = QVBoxLayout()
        tactical_title = QLabel("PAPAN TAKTIK 2D")
        tactical_title.setObjectName("CardTitle")
        tactical_info.addWidget(tactical_title)
        legend = QLabel("● Tim A    ● Tim B    ● Belum pasti")
        legend.setText("<span style='color:#37b9ff'>● Tim A</span> &nbsp; "
                       "<span style='color:#ff6e55'>● Tim B</span> &nbsp; "
                       "<span style='color:#b9afa5'>● ?</span>")
        tactical_info.addWidget(legend)
        self.tactical_mode = QLabel("Analisis video untuk melihat posisi pemain")
        self.tactical_mode.setObjectName("Hint")
        self.tactical_mode.setWordWrap(True)
        tactical_info.addWidget(self.tactical_mode)
        self.btn_calibrate = QPushButton("Kalibrasi 4 sudut")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self.calibrate_court)
        tactical_info.addWidget(self.btn_calibrate)
        self.btn_reset_calibration = QPushButton("Gunakan posisi otomatis")
        self.btn_reset_calibration.setEnabled(False)
        self.btn_reset_calibration.clicked.connect(self.reset_calibration)
        tactical_info.addWidget(self.btn_reset_calibration)
        tactical_info.addStretch()
        tactical_row.addLayout(tactical_info, 1)
        v.addLayout(tactical_row)
        board, mode = draw_tactical()
        self.tactical_view.set_image(frame_image(board))
        body.addWidget(video_card, 1)

        side = QWidget()
        side.setFixedWidth(340)
        s = QVBoxLayout(side)
        s.setContentsMargins(0, 0, 0, 0)
        s.setSpacing(12)

        src = Card("Video")
        self.name_label = QLabel("Belum ada video")
        self.name_label.setObjectName("Muted")
        self.name_label.setWordWrap(True)
        self.meta_label = QLabel("—")
        self.meta_label.setObjectName("Mono")
        src.layout.addWidget(self.name_label)
        src.layout.addWidget(self.meta_label)
        s.addWidget(src)

        status = Card("Status")
        self.status_label = QLabel("Siap")
        self.status_label.setObjectName("Status")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        status.layout.addWidget(self.status_label)
        status.layout.addWidget(self.progress)
        self.btn_review = QPushButton("Tinjau tagging")
        self.btn_review.setEnabled(False)
        self.btn_review.clicked.connect(self.review_identities)
        status.layout.addWidget(self.btn_review)
        self.review_label = QLabel("ID otomatis belum merupakan identitas terverifikasi.")
        self.review_label.setWordWrap(True)
        self.review_label.setObjectName("Hint")
        status.layout.addWidget(self.review_label)
        s.addWidget(status)

        stats = Card("Ringkasan")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        self.stat_ids = self._stat(grid, 0, 0, "FRAGMEN / ID")
        self.stat_active = self._stat(grid, 0, 1, "PEMAIN AKTIF")
        self.stat_ball = self._stat(grid, 1, 0, "CAKUPAN BOLA")
        self.stat_frames = self._stat(grid, 1, 1, "FRAME")
        stats.layout.addLayout(grid)
        s.addWidget(stats)

        ids = Card("ID Aktif")
        self.ids_label = QLabel("—")
        self.ids_label.setObjectName("Mono")
        self.ids_label.setWordWrap(True)
        ids.layout.addWidget(self.ids_label)
        s.addWidget(ids)

        roster = Card("Pemain tersimpan")
        self.roster_card = roster
        self.roster = QListWidget()
        self.roster.setMinimumHeight(160)
        self.roster.setMaximumHeight(250)
        self.roster.setIconSize(QSize(32, 56))
        self.roster.currentItemChanged.connect(self.select_player)
        roster.layout.addWidget(self.roster)
        self.player_name = QLineEdit()
        self.player_name.setPlaceholderText("Nama pemain (opsional)")
        self.player_name.setMaxLength(80)
        self.player_name.returnPressed.connect(self.save_player)
        roster.layout.addWidget(self.player_name)
        self.player_team = QComboBox()
        for team in (0, 1, -1):
            self.player_team.addItem(TEAM_NAMES[team], team)
        roster.layout.addWidget(self.player_team)
        self.player_number = QLineEdit()
        self.player_number.setPlaceholderText("Nomor punggung 0–99 (bila terbaca)")
        self.player_number.setMaxLength(2)
        roster.layout.addWidget(self.player_number)
        self.player_role = QComboBox()
        for role, caption in (("player", "Pemain"), ("goalkeeper", "Kiper"), ("referee", "Wasit · abaikan"), ("other", "Bukan pemain · abaikan")):
            self.player_role.addItem(caption, role)
        roster.layout.addWidget(self.player_role)
        self.btn_save_player = QPushButton("Simpan & verifikasi pemain")
        self.btn_save_player.clicked.connect(self.save_player)
        roster.layout.addWidget(self.btn_save_player)
        self.btn_merge = QPushButton("Gabungkan ID pemain yang sama")
        self.btn_merge.clicked.connect(self.merge_player)
        roster.layout.addWidget(self.btn_merge)
        self.btn_split = QPushButton("Pisahkan ID mulai frame ini")
        self.btn_split.clicked.connect(self.split_identity)
        roster.layout.addWidget(self.btn_split)
        for control in (self.player_name, self.player_team, self.player_number, self.player_role,
                        self.btn_save_player, self.btn_merge, self.btn_split):
            control.setEnabled(False)
        self.saved_label = QLabel("Nama dan pemain tersimpan otomatis per video.")
        self.saved_label.setObjectName("Hint")
        self.saved_label.setWordWrap(True)
        roster.layout.addWidget(self.saved_label)
        s.addWidget(roster)

        self.show_skeleton = QCheckBox("Tampilkan skeleton tubuh")
        self.show_skeleton.setChecked(True)
        self.show_skeleton.toggled.connect(lambda _: self.show_frame(self.current_idx, force=True))
        s.addWidget(self.show_skeleton)

        self.show_court = QCheckBox("Tampilkan area lapangan")
        self.show_court.setEnabled(False)
        self.show_court.toggled.connect(lambda _: self.show_frame(self.current_idx, force=True))
        s.addWidget(self.show_court)
        self.btn_boundary = QPushButton("Koreksi batas area bermain")
        self.btn_boundary.setEnabled(False)
        self.btn_boundary.clicked.connect(self.correct_floor)
        s.addWidget(self.btn_boundary)

        hint = QLabel(
            "Hanya pemain yang titik kakinya berada di lantai biru yang dihitung. "
            "Memori penampilan mencocokkan pemain yang kembali. Seragam sama/tertutup "
            "bisa membuat ID ambigu; pilih pemain untuk mengganti nama, tim, atau menggabungkan ID."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        s.addWidget(hint)
        s.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(365)
        scroll.setWidget(side)
        body.addWidget(scroll)

    def _stat(self, grid, row, col, caption):
        box = QVBoxLayout()
        box.setSpacing(1)
        value = QLabel("—")
        value.setObjectName("StatValue")
        label = QLabel(caption)
        label.setObjectName("StatLabel")
        box.addWidget(value)
        box.addWidget(label)
        wrapper = QWidget()
        wrapper.setLayout(box)
        grid.addWidget(wrapper, row, col)
        return value

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #080d16; color: #dbe7f5; font-family: 'Segoe UI'; font-size: 13px; }
            QLabel#Title { font-size: 26px; font-weight: 700; color: #f4f9ff; }
            QLabel#Subtitle { font-size: 12px; color: #7f93ad; }
            QLabel#Badge { background: #10233b; color: #7dd3fc; border: 1px solid #1e4d6b;
                           border-radius: 9px; padding: 6px 10px; font-size: 10px; font-weight: 600; }
            QFrame#VideoCard { background: #05080e; border: 1px solid #1b2637; border-radius: 16px; }
            QFrame#Card { background: #0d1523; border: 1px solid #1b283c; border-radius: 14px; }
            QLabel#CardTitle { color: #6d819b; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
            QLabel#Muted { color: #94a8c2; font-size: 12px; }
            QLabel#Mono { color: #8ea5c0; font-family: Consolas; font-size: 11px; }
            QLabel#Status { color: #e2f0ff; font-size: 13px; font-weight: 600; }
            QLabel#StatValue { color: #f3f9ff; font-size: 24px; font-weight: 700; }
            QLabel#StatLabel { color: #687d97; font-size: 9px; font-weight: 700; letter-spacing: 0.5px; }
            QLabel#Hint { color: #5e728c; font-size: 11px; line-height: 1.4; }
            QLabel#Time { color: #8ea5c0; font-family: Consolas; font-size: 11px; }
            QPushButton { background: #152238; color: #d7e6f7; border: 1px solid #263852;
                          border-radius: 9px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: #1c2d47; }
            QPushButton:disabled { color: #52647a; background: #101a2a; border-color: #1a273a; }
            QPushButton#Primary { background: #0e7490; border-color: #0891b2; color: #ecfeff; }
            QPushButton#Primary:hover { background: #0891b2; }
            QPushButton#Primary:disabled { background: #123240; border-color: #1c4351; color: #6b8b96; }
            QSlider::groove:horizontal { height: 4px; background: #1b2a40; border-radius: 2px; }
            QSlider::handle:horizontal { background: #38bdf8; width: 14px; height: 14px;
                                         margin: -5px 0; border-radius: 7px; }
            QProgressBar { background: #162336; border: none; border-radius: 4px; height: 6px; }
            QProgressBar::chunk { background: #22d3ee; border-radius: 4px; }
            QCheckBox { color: #94a8c2; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
                                   border: 1px solid #2b3f5b; background: #0d1728; }
            QCheckBox::indicator:checked { background: #0e7490; border-color: #22d3ee; }
            QListWidget, QLineEdit, QComboBox { background: #0a1220; border: 1px solid #263852;
                                              border-radius: 6px; padding: 6px; }
            QListWidget::item { padding: 4px; border-radius: 5px; }
            QListWidget::item:selected { background: #19344b; }
            QScrollArea { border: none; }
        """)

    # ---------- video ----------
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih video futsal", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.webm);;Semua file (*.*)",
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path):
        if self.worker is not None:
            return
        candidate = cv2.VideoCapture(path)
        if not candidate.isOpened():
            candidate.release()
            QMessageBox.critical(self, "Gagal", "Video tidak bisa dibuka.")
            return
        self.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = candidate

        self.video_path = path
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tracks = None
        self.previous_roster = None
        self.roster.clear()
        self.roster_items.clear()
        self.player_name.clear()
        self.player_number.clear()
        self.btn_review.setText("Tinjau tagging")
        self.review_label.setText("ID otomatis belum merupakan identitas terverifikasi.")
        self.saved_label.setText("Nama dan pemain tersimpan otomatis per video.")
        for control in (self.btn_save_player, self.btn_merge, self.btn_split, self.player_name, self.player_team,
                        self.player_number, self.player_role, self.btn_review, self.btn_boundary,
                        self.btn_calibrate, self.btn_reset_calibration):
            control.setEnabled(False)
        self.last_read = -2
        self.current_idx = 0

        self.name_label.setText(os.path.basename(path))
        dur = self.frame_count / max(1.0, self.fps)
        self.meta_label.setText(f"{self.width}×{self.height}  ·  {self.fps:.0f} fps  ·  {dur:.1f} dtk")
        self.slider.setRange(0, max(0, self.frame_count - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.show_court.setEnabled(False)
        self.status_label.setText("Siap dianalisis")
        self.ids_label.setText("—")
        for v in (self.stat_ids, self.stat_active, self.stat_ball, self.stat_frames):
            v.setText("—")
        self.show_frame(0)
        try:
            saved = load_session(path)
            if saved is not None:
                if saved.get("pipeline_version", 0) < 3:
                    self.previous_roster = saved.get("players")
                    self.status_label.setText("Hasil versi lama · jalankan Analisis untuk filter lapangan ketat")
                else:
                    self._apply_tracks(saved, autoplay=False)
                    self.status_label.setText("Hasil tersimpan dimuat")
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            self.status_label.setText("Data tersimpan tidak terbaca; analisis ulang tersedia")
            self.saved_label.setText(str(exc))

    def _read(self, idx):
        if self.cap is None:
            return None
        if idx != self.last_read + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_read = idx
        return frame

    def show_frame(self, idx, force=False):
        if self.cap is None:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        frame = self._read(idx)
        if frame is None:
            return
        self.current_idx = idx
        if self.tracks is not None and idx < len(self.tracks["frames"]):
            selected = self.selected_player_id()
            frame = draw_overlay(frame, self.tracks, idx, self.show_court.isChecked(),
                                 self.show_skeleton.isChecked(), selected)
            fr = self.tracks["frames"][idx]
            active = [p[0] for p in visible_players(self.tracks, idx)]
            self.stat_active.setText(str(len(active)))
            self.ids_label.setText(", ".join(player_label(self.tracks, pid) for pid in active) if active else "Tidak ada")
            self.refresh_roster(set(active))
        else:
            self.stat_active.setText("—")
            self.ids_label.setText("Frame ini belum dianalisis" if self.tracks else "—")
            self.refresh_roster(set())

        board, mode = draw_tactical(self.tracks, idx)
        self.tactical_view.set_image(frame_image(board))
        self.tactical_mode.setText(mode)
        analyzed = self.tracks is not None and idx < len(self.tracks["frames"])
        self.btn_calibrate.setEnabled(analyzed and self.worker is None)
        self.btn_boundary.setEnabled(analyzed and self.worker is None)
        self.btn_reset_calibration.setEnabled(analyzed and bool(self.tracks.get("calibrations")))

        self.view.set_image(frame_image(frame))

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        dur = max(0.0, (self.frame_count - 1) / max(1.0, self.fps))
        now = idx / max(1.0, self.fps)
        self.time_label.setText(f"{int(now // 60):02d}:{int(now % 60):02d} / "
                                f"{int(dur // 60):02d}:{int(dur % 60):02d}")

    def seek(self, idx):
        self.last_read = -2
        self.show_frame(idx)

    def _tick(self):
        if self.frame_count == 0:
            self.stop()
            return
        nxt = self.current_idx + 1
        if nxt >= self.frame_count:
            self.stop()
            return
        before = self.current_idx
        self.show_frame(nxt)
        if self.current_idx == before:  # decoder ended early; do not spin
            self.stop()

    def toggle_play(self):
        if self.cap is None or self.frame_count == 0:
            return
        if self.playing:
            self.stop()
            return
        if self.current_idx >= self.frame_count - 1:
            self.seek(0)
        self.playing = True
        self.btn_play.setText("Jeda")
        self.timer.start(max(15, int(1000 / max(1.0, self.fps))))

    def stop(self):
        self.playing = False
        self.timer.stop()
        if hasattr(self, "btn_play"):
            self.btn_play.setText("Putar")

    # ---------- saved roster and tactics ----------
    def selected_player_id(self):
        item = self.roster.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def build_roster(self):
        self.roster.blockSignals(True)
        self.roster.clear()
        self.roster_items = {}
        for key, player in sorted(self.tracks.get("players", {}).items(), key=lambda pair: int(pair[0])):
            pid = int(key)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, pid)
            if player.get("portrait"):
                pixmap = QPixmap()
                pixmap.loadFromData(base64.b64decode(player["portrait"]))
                item.setIcon(QIcon(pixmap))
            self.roster.addItem(item)
            self.roster_items[pid] = item
        self.roster.blockSignals(False)
        self.refresh_roster(set())
        if self.roster.count():
            self.roster.setCurrentRow(0)

    def refresh_roster(self, active):
        if self.tracks is None:
            return
        for pid, item in self.roster_items.items():
            player = self.tracks["players"][str(pid)]
            team = player_team(self.tracks, pid)
            state = "Aktif" if pid in active else "Di luar frame"
            if pid not in active and (player.get("first_seen", -1) < 0 or self.current_idx < player["first_seen"]):
                state = "Belum terlihat"
            if player.get("excluded"):
                state = "Diabaikan"
            elif player.get("low_support") and not player.get("verified"):
                state = "Deteksi terlalu singkat"
            verified = "Terverifikasi" if player.get("verified") else "Perlu verifikasi"
            caption = f"{player_label(self.tracks, pid)}\n{TEAM_NAMES.get(team, 'Belum pasti')} · {state} · {verified}"
            if item.text() != caption:
                item.setText(caption)
            b, g, r = TEAM_COLORS.get(team, TEAM_COLORS[-1])
            item.setForeground(QColor(r, g, b))

    def select_player(self, item, previous=None):
        enabled = item is not None and self.tracks is not None
        for control in (self.player_name, self.player_team, self.player_number, self.player_role,
                        self.btn_save_player, self.btn_merge, self.btn_split):
            control.setEnabled(enabled)
        if not enabled:
            return
        player = self.tracks["players"][str(item.data(Qt.UserRole))]
        self.player_name.setText(player.get("name", ""))
        self.player_team.setCurrentIndex(self.player_team.findData(player.get("team", -1)))
        self.player_number.setText(player.get("number", ""))
        self.player_role.setCurrentIndex(self.player_role.findData(player.get("role", "player")))
        self.show_frame(self.current_idx, force=True)

    def persist_edits(self):
        try:
            save_session(self.video_path, self.tracks)
            self.saved_label.setText("Tersimpan · nama, tim, ID, dan kalibrasi")
            self.saved_label.setToolTip(str(session_path(self.video_path)))
            return True
        except (OSError, ValueError) as exc:
            self.saved_label.setText("Perubahan belum tersimpan")
            QMessageBox.warning(self, "Gagal menyimpan", str(exc))
            return False

    def save_player(self):
        pid = self.selected_player_id()
        if pid is None or self.worker is not None:
            return
        try:
            verify_player(self.tracks, pid, self.player_number.text(), self.player_team.currentData(),
                          self.player_name.text(), self.player_role.currentData())
        except ValueError as exc:
            QMessageBox.information(self, "Periksa identitas", str(exc))
            return
        self.persist_edits()
        self.refresh_review_status()
        self.show_frame(self.current_idx, force=True)

    def merge_player(self):
        source = self.selected_player_id()
        if source is None or self.worker is not None:
            return
        self.stop()
        targets = [(pid, player_label(self.tracks, pid)) for pid in self.roster_items if pid != source]
        if not targets:
            return
        label, ok = QInputDialog.getItem(self, "Koreksi ID pemain",
                                        f"Gabungkan P{source} ke ID tetap berikut (nama ID tujuan dipakai):",
                                        [label for _, label in targets], 0, False)
        if not ok:
            return
        target = targets[[entry[1] for entry in targets].index(label)][0]
        try:
            merge_players(self.tracks, source, target)
        except ValueError as exc:
            QMessageBox.information(self, "ID tidak digabung", str(exc))
            return
        self.persist_edits()
        self.refresh_review_status()
        self.build_roster()
        self.roster.setCurrentItem(self.roster_items[target])
        self.show_frame(self.current_idx, force=True)

    def refresh_review_status(self):
        if self.tracks is None:
            return
        refresh_review(self.tracks)
        count = len(self.tracks.get("review_queue", []))
        unverified = self.tracks["stats"].get("unverified_ids", 0)
        self.btn_review.setText(f"Tinjau tagging · {count} kandidat")
        self.btn_review.setEnabled(self.worker is None)
        self.review_label.setText(f"{unverified} ID belum diverifikasi. Tanda ? berarti ID sementara.")
        self.stat_ids.setText(str(self.tracks["stats"]["unique_ids"]))

    def review_identities(self):
        if self.tracks is None or self.worker is not None:
            return
        self.stop()
        dialog = IdentityReviewDialog(self.tracks, self)
        dialog.exec()
        if dialog.modified:
            self.persist_edits()
            self.build_roster()
        self.refresh_review_status()
        if dialog.jump:
            pid, frame_index = dialog.jump
            if pid in self.roster_items:
                self.roster.setCurrentItem(self.roster_items[pid])
            self.seek(frame_index)
        else:
            self.show_frame(self.current_idx, force=True)

    def split_identity(self):
        pid = self.selected_player_id()
        if pid is None or self.worker is not None:
            return
        self.stop()
        try:
            new_id = split_player(self.tracks, pid, self.current_idx)
            frame = self._read(self.current_idx)
            if frame is not None:
                from players import portrait
                box = next((p[1:] for p in self.tracks["frames"][self.current_idx]["p"] if p[0] == new_id), None)
                if box:
                    self.tracks["players"][str(new_id)]["portrait"] = portrait(frame, box)
        except ValueError as exc:
            QMessageBox.information(self, "ID belum bisa dipisahkan", str(exc))
            return
        self.persist_edits()
        self.build_roster()
        self.roster.setCurrentItem(self.roster_items[new_id])
        self.refresh_review_status()

    def correct_floor(self):
        if self.tracks is None or self.worker is not None:
            return
        self.stop()
        frame = self._read(self.current_idx)
        if frame is None:
            return
        dialog = BoundaryDialog(frame_image(frame), self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            removed = correct_boundary(self.tracks, self.current_idx, dialog.canvas.points)
        except ValueError as exc:
            QMessageBox.information(self, "Batas belum sesuai", str(exc))
            return
        self.persist_edits()
        self.refresh_review_status()
        self.status_label.setText(f"Batas diperbarui · {removed} deteksi di luar area dihapus")
        self.show_frame(self.current_idx, force=True)

    def calibrate_court(self):
        if self.tracks is None or self.worker is not None:
            return
        self.stop()
        raw = self._read(self.current_idx)
        if raw is None:
            return
        dialog = CalibrationDialog(frame_image(raw), self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            set_calibration(self.tracks, self.current_idx, dialog.canvas.points)
        except ValueError as exc:
            QMessageBox.information(self, "Sudut belum sesuai", str(exc))
            return
        self.persist_edits()
        self.show_frame(self.current_idx, force=True)

    def reset_calibration(self):
        if self.tracks is None or self.worker is not None:
            return
        segment = self.tracks["frames"][self.current_idx].get("segment", 0)
        self.tracks["calibrations"] = [c for c in self.tracks.get("calibrations", []) if c["segment"] != segment]
        self.persist_edits()
        self.show_frame(self.current_idx, force=True)

    # ---------- analysis ----------
    def start_analysis(self):
        if not self.video_path or self.worker is not None:
            return

        limit = None
        if self.fps > 0 and self.frame_count > 180 * self.fps:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Video panjang")
            box.setText("Video ini lebih dari 3 menit. Analisis seluruh video bisa memakan waktu lama.")
            short = box.addButton("2 menit pertama", QMessageBox.AcceptRole)
            full = box.addButton("Semua video", QMessageBox.DestructiveRole)
            box.addButton("Batal", QMessageBox.RejectRole)
            box.setDefaultButton(short)
            box.exec()
            if box.clickedButton() is short:
                limit = int(120 * self.fps)
            elif box.clickedButton() is not full:
                return

        self.stop()
        self.btn_open.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.slider.setEnabled(False)
        self.roster_card.setEnabled(False)
        self.btn_review.setEnabled(False)
        self.btn_boundary.setEnabled(False)
        self.btn_calibrate.setEnabled(False)
        self.btn_reset_calibration.setEnabled(False)
        self.show_court.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText("Memuat model…")
        self.ids_label.setText("—")
        self.stat_ids.setText("—")
        self.stat_ball.setText("—")
        self.stat_frames.setText("—")

        self.worker = AnalyzeThread(self.video_path, limit,
                                    self.tracks.get("players") if self.tracks else self.previous_roster,
                                    self.tracks.get("floor_boundaries") if self.tracks else None)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.status_label.setText(msg)

    def _on_worker_finished(self):
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        worker.deleteLater()
        if self._closing:
            self.close()
            return
        if worker.error:
            self._on_failed(worker.error)
        else:
            self._apply_tracks(worker.result)
            if worker.save_error:
                self.saved_label.setText("Hasil tampil, tetapi belum tersimpan: " + worker.save_error)

    def _apply_tracks(self, tracks, autoplay=True):
        self.tracks = tracks
        if not self.frame_count:
            self.frame_count = tracks["frame_count"]
        self.current_idx = min(self.current_idx, max(0, self.frame_count - 1))
        self.progress.setValue(100)
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.slider.setEnabled(True)
        self.roster_card.setEnabled(True)
        self.show_court.setEnabled(True)
        self.build_roster()
        self.refresh_review_status()
        self.saved_label.setText("Tersimpan otomatis per video")
        self.saved_label.setToolTip(str(session_path(self.video_path)))

        stats = tracks["stats"]
        self.stat_ids.setText(str(stats["unique_ids"]))
        self.stat_frames.setText(str(tracks["frame_count"]))
        coverage = 100.0 * stats["ball_frames"] / max(1, tracks["frame_count"])
        self.stat_ball.setText(f"{coverage:.0f}%")
        secs = round(tracks["frame_count"] / max(1.0, self.fps))
        self.status_label.setText(f"Selesai  •  {secs // 60}:{secs % 60:02d} dianalisis")
        self.show_frame(self.current_idx, force=True)
        if autoplay:
            self.seek(0)
            self.toggle_play()

    def _on_failed(self, text):
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.slider.setEnabled(True)
        self.roster_card.setEnabled(True)
        self.btn_review.setEnabled(self.tracks is not None)
        self.status_label.setText("Analisis gagal")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Analisis gagal")
        box.setText("Pipeline gagal dijalankan.")
        box.setDetailedText(text)
        box.exec()

    def closeEvent(self, event):
        self.stop()
        if self.worker is not None:
            self._closing = True
            self.worker.requestInterruption()
            self.status_label.setText("Menghentikan analisis…")
            event.ignore()
            return
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Futsal Vision")
    window = MainWindow()
    window.show()
    reference = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "1031(2).mp4")
    if os.path.isfile(reference):
        window.load_video(reference)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
