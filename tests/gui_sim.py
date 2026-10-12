"""Optional PyQt simulation check; run with QT_QPA_PLATFORM=offscreen.

This is deliberately separate from the headless pytest suite. It exercises
the real window, five synthetic workers, alert log creation, and manual reset.
"""
import multiprocessing as mp
from pathlib import Path
import tempfile
import time


def run():
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    import main
    from ui.main_window import MainWindow

    alerts, observations, failures = [], [], []
    original_show = MainWindow.show_alert
    original_exec = QApplication.exec_
    original_log = main.ALERT_LOG

    def observe_alert(window, alert):
        alerts.append(alert["level"])
        original_show(window, alert)

    def checked_exec(app):
        deadline = time.monotonic() + 20.0

        def verify_reset():
            try:
                window = app.activeWindow()
                assert window.alert_label.text() == "No active alert"
                observations.append("manual reset")
            except Exception as exc:
                failures.append(repr(exc))
            app.quit()

        def wait_for_l2():
            if alerts == [1, 2]:
                try:
                    window = app.activeWindow()
                    assert window is not None, "No window"
                    assert window.video_label.pixmap() is not None, "No video"
                    assert window.depth_widget.level == "NEAR"
                    assert all(ts > 0 for ts in window._heartbeats.values())
                    assert window.status_label.text().count("SIMULATION") == 5, (
                        window.status_label.text())
                    assert "EMERGENCY" in window.alert_label.text()
                    observations.append("video, depth, heartbeats, L1 and L2")
                    window.manual_events.append({"demo_key": "R"})
                    QTimer.singleShot(700, verify_reset)
                except Exception as exc:
                    failures.append(repr(exc))
                    app.quit()
            elif time.monotonic() >= deadline:
                failures.append("No L2 before deadline; alerts=" + repr(alerts))
                app.quit()
            else:
                QTimer.singleShot(100, wait_for_l2)

        QTimer.singleShot(100, wait_for_l2)
        return original_exec()

    try:
        MainWindow.show_alert = observe_alert
        QApplication.exec_ = checked_exec
        with tempfile.TemporaryDirectory(prefix="jetson-synthetic-") as temp:
            main.ALERT_LOG = str(Path(temp) / "alerts" / "alerts.log")
            assert main.main(["--sim"]) == 0
            assert not failures, failures
            assert len(observations) == 2, observations
            assert not mp.active_children(), "Workers survived GUI shutdown"
            lines = Path(main.ALERT_LOG).read_text().splitlines()
            assert len(lines) == 2
            assert "\tL1\t" in lines[0] and "\tL2\t" in lines[1]
    finally:
        MainWindow.show_alert = original_show
        QApplication.exec_ = original_exec
        main.ALERT_LOG = original_log
    print("GUI SIM: OK; " + "; ".join(observations)
          + "; log written; all workers stopped")


if __name__ == "__main__":
    run()
