import os
import sys
import traceback

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)

from vision import analyze_video, draw_overlay


class VideoView(QWidget):
    def __init__(self):
        super().__init__()
        self._image = None
        self.setMinimumSize(640, 360)
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


class AnalyzeThread(QThread):
    progress = Signal(int, str)

    def __init__(self, path, max_frames=None):
        super().__init__()
        self.path = path
        self.max_frames = max_frames
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = analyze_video(
                self.path,
                lambda i, total, msg: self.progress.emit(
                    int(min(100, i * 100 / max(1, total))), msg
                ),
                max_frames=self.max_frames,
            )
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
        self.worker = None
        self.playing = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_style()

        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)

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
        subtitle = QLabel("Pelacakan pemain di lapangan biru + bola")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        head.addLayout(title_box)
        head.addStretch()
        badge = QLabel("NATIVE  •  YOLO11s  •  BoT-SORT ReID")
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
        body.addWidget(video_card, 1)

        side = QWidget()
        side.setFixedWidth(310)
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
        s.addWidget(status)

        stats = Card("Ringkasan")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        self.stat_ids = self._stat(grid, 0, 0, "ID UNIK")
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

        self.show_court = QCheckBox("Tampilkan area lapangan")
        self.show_court.setEnabled(False)
        self.show_court.toggled.connect(lambda _: self.show_frame(self.current_idx, force=True))
        s.addWidget(self.show_court)

        hint = QLabel(
            "Hanya pemain yang titik kakinya berada di lantai biru yang dihitung. "
            "ID dijaga oleh ReID + kompensasi gerak kamera."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        s.addWidget(hint)
        s.addStretch()

        body.addWidget(side)

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
        """)

    # ---------- video ----------
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih video futsal", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.webm);;Semua file (*.*)",
        )
        if not path:
            return
        self.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Gagal", "Video tidak bisa dibuka.")
            return

        self.video_path = path
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tracks = None
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
            frame = draw_overlay(frame, self.tracks, idx, self.show_court.isChecked())
            fr = self.tracks["frames"][idx]
            active = [str(p[0]) for p in fr["p"]]
            self.stat_active.setText(str(len(active)))
            self.ids_label.setText(", ".join(f"P{x}" for x in active) if active else "Tidak ada")
        else:
            self.stat_active.setText("—")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.view.set_image(image)

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
        self.show_court.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText("Memuat model…")
        self.ids_label.setText("—")
        self.stat_ids.setText("—")
        self.stat_ball.setText("—")
        self.stat_frames.setText("—")

        self.worker = AnalyzeThread(self.video_path, limit)
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
        if worker.error:
            self._on_failed(worker.error)
        else:
            self._apply_tracks(worker.result)

    def _apply_tracks(self, tracks):
        self.tracks = tracks
        if not self.frame_count:
            self.frame_count = tracks["frame_count"]
        self.current_idx = min(self.current_idx, max(0, self.frame_count - 1))
        self.progress.setValue(100)
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.show_court.setEnabled(True)

        stats = tracks["stats"]
        self.stat_ids.setText(str(stats["unique_ids"]))
        self.stat_frames.setText(str(tracks["frame_count"]))
        coverage = 100.0 * stats["ball_frames"] / max(1, tracks["frame_count"])
        self.stat_ball.setText(f"{coverage:.0f}%")
        secs = round(tracks["frame_count"] / max(1.0, self.fps))
        self.status_label.setText(f"Selesai  •  {secs // 60}:{secs % 60:02d} dianalisis")
        self.show_frame(self.current_idx, force=True)
        self.seek(0)
        self.toggle_play()

    def _on_failed(self, text):
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
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
            self.worker.terminate()
            self.worker.wait(1500)
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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
