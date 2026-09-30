import sys
import re
import subprocess
from pathlib import Path

import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


BASE = Path(__file__).resolve().parent

RM_FILE = BASE / "video_test" / "rx_video_rm.mp4"
OFDM_FILE = BASE / "video_test" / "rx_video_ofdm.mp4"
RAN_FILE = BASE / "Ran_ofdm.py"
RUN_SCRIPT = BASE / "prepare_and_run.sh"


class VideoPlayer(QWidget):

    def __init__(self, title):
        super().__init__()

        layout = QVBoxLayout(self)

        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)

        self.stats = QLabel("")
        self.stats.setAlignment(Qt.AlignCenter)
        self.stats.setStyleSheet(
            "font-weight: bold;"
            "font-size: 13px;"
            "padding: 3px;"
        )

        self.screen = QLabel("No video")
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setMinimumSize(300, 220)
        self.screen.setStyleSheet(
            "background:#000;"
            "color:#ddd;"
            "border:1px solid #555;"
        )

        layout.addWidget(self.title)
        layout.addWidget(self.stats)
        layout.addWidget(self.screen)

        self.cap = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)


    def clear(self):

        self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.screen.clear()
        self.screen.setText("Waiting for new result...")
        self.stats.setText("")


    def play(self, filename):

        self.timer.stop()

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(str(filename))

        if not self.cap.isOpened():
            self.screen.setText(
                "Cannot open:\n" + str(filename)
            )
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        self.timer.start(
            max(1, int(1000 / fps))
        )


    def next_frame(self):

        if self.cap is None:
            return

        ok, frame = self.cap.read()

        if not ok:

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ok, frame = self.cap.read()

            if not ok:
                return

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, c = frame.shape

        image = QImage(
            frame.data,
            w,
            h,
            w * c,
            QImage.Format_RGB888
        ).copy()

        pix = QPixmap.fromImage(image)

        self.screen.setPixmap(
            pix.scaled(
                self.screen.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )


class Runner(QThread):

    output = pyqtSignal(str)
    success = pyqtSignal()
    error = pyqtSignal(str)

    # branch, decoded, exact, BER
    stats = pyqtSignal(str, str, str, str)

    def __init__(self, input_video, noise):
        super().__init__()

        self.input_video = str(input_video)
        self.noise = float(noise)


    def set_noise(self):

        text = RAN_FILE.read_text()

        pattern = (
            r"self\.noise_volt\s*=\s*"
            r"noise_volt\s*=\s*"
            r"[-+0-9.eE]+"
        )

        replacement = (
            f"self.noise_volt = "
            f"noise_volt = {self.noise}"
        )

        text, count = re.subn(
            pattern,
            replacement,
            text,
            count=1
        )

        if count != 1:
            raise RuntimeError(
                "noise_volt not found in Ran_ofdm.py"
            )

        RAN_FILE.write_text(text)


    def run(self):

        try:

            self.set_noise()

            self.output.emit(
                "INPUT VIDEO:"
            )

            self.output.emit(
                self.input_video
            )

            self.output.emit(
                f"NOISE: {self.noise}"
            )

            self.output.emit(
                "=============================="
            )

            process = subprocess.Popen(
                [
                    "bash",
                    str(RUN_SCRIPT),
                    self.input_video
                ],
                cwd=str(BASE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:

                clean = line.rstrip()

                self.output.emit(
                    clean
                )

                # Example:
                # [VIDEO_RX] OFDM: decoded=24/229 exact=0 BER=0.00285 incomplete=[...]

                match = re.search(
                    r"\[VIDEO_RX\]\s+"
                    r"(RM|OFDM):\s+"
                    r"decoded=([^\s]+)\s+"
                    r"exact=([^\s]+)\s+"
                    r"BER=([^\s]+)",
                    clean
                )

                if match:

                    branch = match.group(1)
                    decoded = match.group(2)
                    exact = match.group(3)
                    ber = match.group(4)

                    self.stats.emit(
                        branch,
                        decoded,
                        exact,
                        ber
                    )

            code = process.wait()

            if code != 0:

                raise RuntimeError(
                    f"Simulation failed: {code}"
                )

            if not RM_FILE.exists():
                raise RuntimeError(
                    "RM output was not created"
                )

            if not OFDM_FILE.exists():
                raise RuntimeError(
                    "OFDM output was not created"
                )

            self.success.emit()

        except Exception as e:

            self.error.emit(
                str(e)
            )


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.input_path = None
        self.runner = None

        self.setWindowTitle(
            "OFDM Video Transmission Simulator"
        )

        self.resize(
            1200,
            760
        )

        root = QVBoxLayout(self)


        # -------------------------
        # Input button
        # -------------------------

        self.select_button = QPushButton(
            "Select Input Video"
        )

        self.select_button.clicked.connect(
            self.select_video
        )

        root.addWidget(
            self.select_button
        )


        # -------------------------
        # Three videos
        # -------------------------

        videos = QHBoxLayout()

        self.input_player = VideoPlayer(
            "Input Video"
        )

        self.input_player.stats.setText(
            "Original source"
        )

        self.rm_player = VideoPlayer(
            "RM Received"
        )

        self.ofdm_player = VideoPlayer(
            "OFDM Received"
        )

        videos.addWidget(
            self.input_player
        )

        videos.addWidget(
            self.rm_player
        )

        videos.addWidget(
            self.ofdm_player
        )

        root.addLayout(videos)


        # -------------------------
        # Controls
        # -------------------------

        controls = QHBoxLayout()

        controls.addWidget(
            QLabel("Noise")
        )

        self.noise = QDoubleSpinBox()

        self.noise.setRange(
            0.0,
            1.0
        )

        self.noise.setDecimals(2)

        self.noise.setSingleStep(
            0.05
        )

        self.noise.setValue(
            0.0
        )

        controls.addWidget(
            self.noise
        )


        self.run_button = QPushButton(
            "Run Simulation"
        )

        self.run_button.clicked.connect(
            self.run_simulation
        )

        controls.addWidget(
            self.run_button
        )

        root.addLayout(
            controls
        )


        # -------------------------
        # Log
        # -------------------------

        self.log = QTextEdit()

        self.log.setReadOnly(
            True
        )

        self.log.setMaximumHeight(
            300
        )

        root.addWidget(
            self.log
        )


    def select_video(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )

        if not filename:
            return

        self.input_path = filename

        self.input_player.play(
            filename
        )

        self.log.append(
            "Selected:"
        )

        self.log.append(
            filename
        )


    def run_simulation(self):

        if not self.input_path:

            QMessageBox.warning(
                self,
                "Input",
                "Select a video first."
            )

            return

        # Important:
        # stop showing old results immediately.
        self.rm_player.clear()
        self.ofdm_player.clear()

        self.run_button.setEnabled(
            False
        )

        self.select_button.setEnabled(
            False
        )

        self.log.clear()

        self.runner = Runner(
            self.input_path,
            self.noise.value()
        )

        self.runner.output.connect(
            self.log.append
        )

        self.runner.stats.connect(
            self.update_stats
        )

        self.runner.success.connect(
            self.finished_ok
        )

        self.runner.error.connect(
            self.finished_error
        )

        self.runner.start()


    def update_stats(
        self,
        branch,
        decoded,
        exact,
        ber
    ):

        try:
            ber_value = float(ber)

            ber_text = (
                f"{ber_value:.8f}"
            )

        except Exception:
            ber_text = ber

        text = (
            f"BER: {ber_text}    |    "
            f"Decoded: {decoded}    |    "
            f"Exact: {exact}"
        )

        if branch == "RM":

            self.rm_player.stats.setText(
                text
            )

        elif branch == "OFDM":

            self.ofdm_player.stats.setText(
                text
            )


    def finished_ok(self):

        self.run_button.setEnabled(
            True
        )

        self.select_button.setEnabled(
            True
        )

        self.input_player.play(
            self.input_path
        )

        self.rm_player.play(
            RM_FILE
        )

        self.ofdm_player.play(
            OFDM_FILE
        )

        self.log.append(
            ""
        )

        self.log.append(
            "=== DONE ==="
        )

        self.log.append(
            "New RM/OFDM results loaded."
        )


    def finished_error(self, text):

        self.run_button.setEnabled(
            True
        )

        self.select_button.setEnabled(
            True
        )

        self.log.append(
            "ERROR: " + text
        )

        QMessageBox.critical(
            self,
            "Simulation Error",
            text
        )


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(
    app.exec_()
)
