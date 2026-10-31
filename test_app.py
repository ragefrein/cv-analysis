"""Offscreen UI checks: editing, reload, seeking, and calibration click mapping."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt, QThread
from PySide6.QtGui import QFontDatabase, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import storage
from app import CalibrationView, IdentityReviewDialog, MainWindow
from test_identity import tracklet_fixture
from test_vision import analysis_fixture


@pytest.fixture(scope="module")
def application():
    application = QApplication.instance() or QApplication([])
    # The Windows offscreen platform has no system font enumeration.
    font = "C:/Windows/Fonts/segoeui.ttf"
    if os.path.isfile(font):
        QFontDatabase.addApplicationFont(font)
    return application


def test_player_edit_and_reopen(application, tmp_path, monkeypatch):
    def fail_dialog(parent, title, message):
        pytest.fail(f"Unexpected dialog: {title}: {message}")

    monkeypatch.setattr("app.QMessageBox.warning", fail_dialog)
    monkeypatch.setattr("app.QMessageBox.critical", fail_dialog)
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path / "sessions")
    path = tmp_path / "match.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 180))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((180, 320, 3), (254, 209, 195), np.uint8))
    writer.release()
    tracks = analysis_fixture()
    tracks.update(width=320, height=180, fps=25)
    tracks["stats"].update(ball_frames=0, court_frames=2)
    for fr in tracks["frames"]:
        fr["b"] = None
    window = MainWindow()
    window.load_video(str(path))
    window._apply_tracks(tracks, autoplay=False)
    assert window.roster.count() == 2
    window.player_name.setText("Andi Pratama")
    window.player_team.setCurrentIndex(window.player_team.findData(1))
    window.save_player()
    assert storage.load_session(path)["players"]["0"]["name"] == "Andi Pratama"
    window.close()

    reopened = MainWindow()
    reopened.load_video(str(path))
    assert reopened.tracks["players"]["0"]["name"] == "Andi Pratama"
    assert reopened.tracks["players"]["0"]["team"] == 1
    reopened.seek(3)  # outside the analyzed prefix: no stale active players
    assert reopened.stat_active.text() == "—"
    assert reopened.ids_label.text() == "Frame ini belum dianalisis"
    assert not reopened.btn_calibrate.isEnabled()
    reopened.close()


def test_calibration_clicks_account_for_letterboxing(application):
    image = QImage(200, 100, QImage.Format_RGB888)
    image.fill(Qt.black)
    canvas = CalibrationView(image)
    canvas.resize(600, 600)
    canvas.show()
    application.processEvents()
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(150, 225))
    assert np.allclose(canvas.points[0], [50, 25])
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(10, 10))
    assert len(canvas.points) == 1
    QTest.mouseClick(canvas, Qt.RightButton, pos=QPoint(150, 225))
    assert canvas.points == []
    canvas.close()


def test_close_requests_worker_interruption_without_error_dialog(application):
    class Worker(QThread):
        error = "cancelled"

        def run(self):
            while not self.isInterruptionRequested():
                self.msleep(5)

    window = MainWindow()
    window.show()
    worker = Worker()
    window.worker = worker
    worker.finished.connect(window._on_worker_finished)
    worker.start()
    window.close()
    assert window._closing
    for _ in range(20):
        QTest.qWait(25)
        if window.worker is None:
            break
    assert window.worker is None
    assert not window.isVisible()


def test_review_dialog_saves_distinct_identity_decision(application):
    tracks = tracklet_fixture(second_segment=1)
    window = MainWindow()
    dialog = IdentityReviewDialog(tracks, window)
    assert len(dialog.links) == 1
    dialog.reject_link()
    assert dialog.modified
    assert dialog.links == []
    assert tracks["review_decisions"][0]["action"] == "different"
    dialog.close()
    window.close()


def test_review_dialog_can_merge_confirmed_fragment(application):
    tracks = tracklet_fixture(second_segment=1)
    window = MainWindow()
    dialog = IdentityReviewDialog(tracks, window)
    dialog.accept_link()
    assert dialog.modified
    assert len(tracks["players"]) == 1
    assert tracks["review_decisions"][0]["action"] == "same"
    dialog.close()
    window.close()


def test_old_cache_requires_reanalysis_and_cannot_seed_different_video(application, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path / "sessions")
    videos = [tmp_path / "old.avi", tmp_path / "different.avi"]
    for path in videos:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 180))
        assert writer.isOpened()
        writer.write(np.full((180, 320, 3), 100, np.uint8))
        writer.release()
    old_tracks = analysis_fixture()
    old_tracks["pipeline_version"] = 2
    storage.save_session(videos[0], old_tracks)
    window = MainWindow()
    window.load_video(str(videos[0]))
    assert window.tracks is None
    assert window.previous_roster["0"]["name"] == "Andi"
    window.load_video(str(videos[1]))
    assert window.previous_roster is None
    assert window.tracks is None
    window.close()
