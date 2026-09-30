import os
import sys

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets


DEFAULT_PACKET_LEN = 274
WHITEN_SEED_BASE = 2026
STANDALONE_ID_BYTES = 1
COMBO_ID_BYTES = 2


def prepare_image_packets(
    input_path,
    output_folder,
    image_width,
    image_height,
    packet_len
):
    """Resize an image and generate standalone and COMBO packet files."""
    packet_len = int(packet_len)
    image_width = int(image_width)
    image_height = int(image_height)

    if packet_len % 2 != 0:
        raise ValueError(
            "Packet Length must be even so COMB can be split equally."
        )
    if packet_len <= 2 * COMBO_ID_BYTES:
        raise ValueError("Packet Length is too small for COMB packet IDs.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    os.makedirs(output_folder, exist_ok=True)

    with Image.open(input_path) as source_image:
        image = source_image.convert("RGB").resize(
            (image_width, image_height)
        )

    reference_path = os.path.join(output_folder, "tx_image_rgb.png")
    image.save(reference_path)
    image_bytes = np.asarray(image, dtype=np.uint8).reshape(-1)
    image_size = len(image_bytes)

    generated = {
        "reference": reference_path,
        "normal_raw": None,
        "normal_whitened": None,
        "combo_raw": None,
        "combo_whitened": None,
        "normal_packets": 0,
        "combo_packets": 0,
        "normal_error": None
    }

    # Standalone RM/OFDM packet: [1-byte ID][payload].  ID 0xFF is
    # reserved for the warm-up packet, leaving IDs 0..254 for image data.
    normal_payload_len = packet_len - STANDALONE_ID_BYTES
    normal_packets = (
        image_size + normal_payload_len - 1
    ) // normal_payload_len
    generated["normal_packets"] = normal_packets

    if normal_packets <= 255:
        normal_raw = np.zeros(
            (normal_packets + 1, packet_len),
            dtype=np.uint8
        )
        normal_whitened = np.zeros_like(normal_raw)
        normal_raw[0, 0] = 0xFF
        normal_whitened[0, 0] = 0xFF

        for packet_id in range(normal_packets):
            start = packet_id * normal_payload_len
            payload = image_bytes[start:start + normal_payload_len]
            valid_length = len(payload)

            normal_raw[packet_id + 1, 0] = packet_id
            normal_raw[
                packet_id + 1,
                1:1 + valid_length
            ] = payload

            rng = np.random.default_rng(WHITEN_SEED_BASE + packet_id)
            mask = rng.integers(
                0,
                256,
                size=normal_payload_len,
                dtype=np.uint8
            )
            padded_payload = normal_raw[packet_id + 1, 1:].copy()
            normal_whitened[packet_id + 1, 0] = packet_id
            normal_whitened[packet_id + 1, 1:] = padded_payload ^ mask

        normal_raw_path = os.path.join(output_folder, "image_packets.bin")
        normal_whitened_path = os.path.join(
            output_folder,
            "image_packets_whitened.bin"
        )
        normal_raw.tofile(normal_raw_path)
        normal_whitened.tofile(normal_whitened_path)
        generated["normal_raw"] = normal_raw_path
        generated["normal_whitened"] = normal_whitened_path
    else:
        minimum_payload = (image_size + 254) // 255
        minimum_packet_len = minimum_payload + STANDALONE_ID_BYTES
        if minimum_packet_len % 2 != 0:
            minimum_packet_len += 1
        generated["normal_error"] = (
            f"RM/OFDM requires {normal_packets} packets, exceeding the "
            "255-packet limit. Set Packet Length to at least "
            f"{minimum_packet_len}."
        )

    # COMBO packet: [RM branch][OFDM branch].  Each branch contains a
    # two-byte big-endian ID followed by an identical image payload.
    branch_len = packet_len // 2
    combo_payload_len = branch_len - COMBO_ID_BYTES
    combo_packets = (
        image_size + combo_payload_len - 1
    ) // combo_payload_len
    generated["combo_packets"] = combo_packets

    if combo_packets > 65535:
        raise ValueError("Too many packets for the two-byte COMB packet ID.")

    combo_raw = np.zeros(
        (combo_packets + 1, packet_len),
        dtype=np.uint8
    )
    combo_whitened = np.zeros_like(combo_raw)
    combo_raw[0, 0:2] = 0xFF
    combo_raw[0, branch_len:branch_len + 2] = 0xFF
    combo_whitened[0] = combo_raw[0]

    for packet_id in range(combo_packets):
        start = packet_id * combo_payload_len
        payload = image_bytes[start:start + combo_payload_len]
        valid_length = len(payload)
        packet_id_bytes = np.array(
            [(packet_id >> 8) & 0xFF, packet_id & 0xFF],
            dtype=np.uint8
        )

        raw_branch = np.zeros(branch_len, dtype=np.uint8)
        raw_branch[:COMBO_ID_BYTES] = packet_id_bytes
        raw_branch[
            COMBO_ID_BYTES:COMBO_ID_BYTES + valid_length
        ] = payload

        rng = np.random.default_rng(WHITEN_SEED_BASE + packet_id)
        mask = rng.integers(
            0,
            256,
            size=combo_payload_len,
            dtype=np.uint8
        )
        whitened_branch = raw_branch.copy()
        whitened_branch[COMBO_ID_BYTES:] ^= mask

        row = packet_id + 1
        combo_raw[row, :branch_len] = raw_branch
        combo_raw[row, branch_len:] = raw_branch
        combo_whitened[row, :branch_len] = whitened_branch
        combo_whitened[row, branch_len:] = whitened_branch

    combo_raw_path = os.path.join(
        output_folder,
        "image_packets_combo.bin"
    )
    combo_whitened_path = os.path.join(
        output_folder,
        "image_packets_combo_whitened.bin"
    )
    combo_raw.tofile(combo_raw_path)
    combo_whitened.tofile(combo_whitened_path)
    generated["combo_raw"] = combo_raw_path
    generated["combo_whitened"] = combo_whitened_path

    return generated


class RadioImageUI(QtWidgets.QWidget):
    """Launch a GNU Radio image experiment and display its result images."""

    RESULT_TIMEOUT_MS = 300_000
    PREVIEW_SIZE = 280

    def __init__(self):
        super().__init__()

        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.picture_dir = os.path.join(self.project_dir, "picture")

        self.process = None
        self.process_tail = ""
        self.run_completed = False
        self.output_snapshot = {}
        self.current_outputs = {}
        self.generated_files = {}
        self.generated_config = None

        self.setWindowTitle("RM / OFDM Image Transmission")
        self.resize(1280, 760)

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.check_result_files)

        self.timeout_timer = QtCore.QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self.handle_timeout)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.build_settings_panel(), 1)
        main_layout.addLayout(self.build_results_panel(), 3)

        self.mode.currentTextChanged.connect(self.update_mode_view)
        self.width_box.valueChanged.connect(self.update_reference_size_info)
        self.height_box.valueChanged.connect(self.update_reference_size_info)
        self.width_box.valueChanged.connect(self.invalidate_generated_bins)
        self.height_box.valueChanged.connect(self.invalidate_generated_bins)
        self.packet_len_box.valueChanged.connect(self.invalidate_generated_bins)
        self.reference_image_path.textChanged.connect(
            self.update_reference_size_info
        )
        self.reference_image_path.textChanged.connect(
            self.invalidate_generated_bins
        )
        self.bin_output_dir.textChanged.connect(self.invalidate_generated_bins)
        self.generate_btn.clicked.connect(self.generate_bin_files)
        self.start_btn.clicked.connect(self.start_transmission)
        self.update_mode_view(self.mode.currentText())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_settings_panel(self):
        settings_box = QtWidgets.QGroupBox("Transmission Settings")
        form = QtWidgets.QFormLayout(settings_box)

        self.reference_image_path, image_browse = self.create_path_input()
        image_browse.clicked.connect(self.browse_reference_image)
        form.addRow(
            "Input Image:",
            self.path_row(self.reference_image_path, image_browse)
        )

        self.bin_output_dir, bin_dir_browse = self.create_path_input(
            self.picture_dir
        )
        bin_dir_browse.clicked.connect(self.browse_bin_output_directory)
        form.addRow(
            "BIN Output Directory:",
            self.path_row(self.bin_output_dir, bin_dir_browse)
        )

        self.width_box = QtWidgets.QSpinBox()
        self.width_box.setRange(1, 8192)
        self.width_box.setValue(128)
        form.addRow("Image Width:", self.width_box)

        self.height_box = QtWidgets.QSpinBox()
        self.height_box.setRange(1, 8192)
        self.height_box.setValue(128)
        form.addRow("Image Height:", self.height_box)

        self.packet_len_box = QtWidgets.QSpinBox()
        self.packet_len_box.setRange(6, 65536)
        self.packet_len_box.setValue(DEFAULT_PACKET_LEN)
        form.addRow("Packet Length:", self.packet_len_box)

        self.generate_btn = QtWidgets.QPushButton("Generate BIN Files")
        form.addRow(self.generate_btn)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["RM", "OFDM", "COMB"])
        form.addRow("Mode:", self.mode)

        self.selected_bin_label = QtWidgets.QLabel("Selected BIN: not generated")
        self.selected_bin_label.setWordWrap(True)
        self.selected_bin_label.setStyleSheet("color: #555;")
        form.addRow(self.selected_bin_label)

        self.gr_file_path, gr_browse = self.create_path_input(
            os.path.join(self.project_dir, "rx_ran_ofdm.py")
        )
        gr_browse.clicked.connect(self.browse_gnuradio_file)
        form.addRow(
            "GNU Radio Python File:",
            self.path_row(self.gr_file_path, gr_browse)
        )

        self.rm_output_path, rm_browse = self.create_path_input(
            os.path.join(self.picture_dir, "rx_rm.png")
        )
        rm_browse.clicked.connect(
            lambda: self.browse_output_file(self.rm_output_path, "RM")
        )
        self.rm_path_label = QtWidgets.QLabel("RM Result File:")
        self.rm_path_row = self.path_row(self.rm_output_path, rm_browse)
        form.addRow(self.rm_path_label, self.rm_path_row)

        self.ofdm_output_path, ofdm_browse = self.create_path_input(
            os.path.join(self.picture_dir, "rx_ofdm.png")
        )
        ofdm_browse.clicked.connect(
            lambda: self.browse_output_file(self.ofdm_output_path, "OFDM")
        )
        self.ofdm_path_label = QtWidgets.QLabel("OFDM Result File:")
        self.ofdm_path_row = self.path_row(self.ofdm_output_path, ofdm_browse)
        form.addRow(self.ofdm_path_label, self.ofdm_path_row)

        self.start_btn = QtWidgets.QPushButton("Start Transmission")
        form.addRow(self.start_btn)

        path_note = QtWidgets.QLabel(
            "RM/OFDM automatically uses image_packets_whitened.bin; COMB "
            "uses image_packets_combo_whitened.bin."
        )
        path_note.setWordWrap(True)
        path_note.setStyleSheet("color: #666;")
        form.addRow(path_note)

        return settings_box

    def build_results_panel(self):
        results = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Transmission Results")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        results.addWidget(title)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        results.addWidget(self.status)

        image_layout = QtWidgets.QHBoxLayout()

        self.tx_result, self.tx_image, self.tx_info = self.create_result_box(
            "Reference"
        )
        self.rm_result, self.rm_image, self.rm_info = self.create_result_box(
            "RM Result"
        )
        self.ofdm_result, self.ofdm_image, self.ofdm_info = self.create_result_box(
            "OFDM Result"
        )

        image_layout.addWidget(self.tx_result)
        image_layout.addWidget(self.rm_result)
        image_layout.addWidget(self.ofdm_result)
        results.addLayout(image_layout)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setPlaceholderText("GNU Radio output")
        results.addWidget(self.log_box, 1)

        return results

    @staticmethod
    def create_path_input(initial_text=""):
        line_edit = QtWidgets.QLineEdit(initial_text)
        browse_button = QtWidgets.QPushButton("Browse...")
        return line_edit, browse_button

    @staticmethod
    def path_row(line_edit, browse_button):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_button)
        return widget

    def create_result_box(self, title):
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)

        image_label = QtWidgets.QLabel("No image")
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        image_label.setStyleSheet(
            "border: 1px solid #aaa; background: #f5f5f5;"
        )

        info_label = QtWidgets.QLabel("Size: -\nBER: -\nPSNR: -")
        info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        layout.addWidget(image_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(info_label)
        return group, image_label, info_label

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def browse_gnuradio_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select GNU Radio Python File",
            self.gr_file_path.text() or self.project_dir,
            "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.gr_file_path.setText(path)

    def browse_bin_output_directory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select BIN Output Directory",
            self.bin_output_dir.text() or self.picture_dir
        )
        if path:
            self.bin_output_dir.setText(path)

    def browse_reference_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            self.reference_image_path.text() or self.project_dir,
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if not path:
            return

        self.reference_image_path.setText(path)

    def browse_output_file(self, line_edit, branch):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Select {branch} Result File",
            line_edit.text() or self.picture_dir,
            "PNG Image (*.png);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def update_reference_size_info(self, *args):
        path = self.reference_image_path.text().strip()
        if not path or not os.path.isfile(path):
            return
        try:
            with Image.open(path) as source_image:
                width, height = source_image.size
                preview_image = source_image.convert("RGB").resize(
                    (self.width_box.value(), self.height_box.value())
                )
            self.show_pil_image(self.tx_image, preview_image)
        except (OSError, ValueError):
            return
        self.tx_info.setText(
            f"Source: {width} x {height}\n"
            f"Preview: {self.width_box.value()} x "
            f"{self.height_box.value()}\n"
            "BER: -\nPSNR: -"
        )

    def current_generation_config(self):
        image_path = self.reference_image_path.text().strip()
        output_dir = self.bin_output_dir.text().strip()
        return (
            os.path.abspath(image_path) if image_path else "",
            os.path.abspath(output_dir) if output_dir else "",
            self.width_box.value(),
            self.height_box.value(),
            self.packet_len_box.value()
        )

    def invalidate_generated_bins(self, *args):
        if self.generated_config is None:
            return
        if self.generated_config != self.current_generation_config():
            self.generated_files = {}
            self.generated_config = None
            self.selected_bin_label.setText(
                "Selected BIN: settings changed; generate again"
            )
            self.selected_bin_label.setStyleSheet("color: #b06a00;")

    def selected_bin_path(self):
        if self.mode.currentText() == "COMB":
            return self.generated_files.get("combo_whitened")
        return self.generated_files.get("normal_whitened")

    def refresh_selected_bin_label(self):
        selected_path = self.selected_bin_path()
        if selected_path:
            self.selected_bin_label.setText(f"Selected BIN: {selected_path}")
            self.selected_bin_label.setStyleSheet("color: #206020;")
        elif self.generated_config is not None:
            reason = self.generated_files.get("normal_error")
            text = reason or "No BIN is available for the selected mode."
            self.selected_bin_label.setText(f"Selected BIN: unavailable\n{text}")
            self.selected_bin_label.setStyleSheet("color: #b00020;")
        else:
            self.selected_bin_label.setText("Selected BIN: not generated")
            self.selected_bin_label.setStyleSheet("color: #555;")

    def generate_bin_files(self):
        image_path = self.reference_image_path.text().strip()
        output_dir = self.bin_output_dir.text().strip()

        if not image_path or not os.path.isfile(image_path):
            self.show_error("Select a valid input image first.")
            return
        if not output_dir:
            self.show_error("Select a BIN output directory.")
            return

        self.generate_btn.setEnabled(False)
        self.status.setStyleSheet("")
        self.status.setText("Generating BIN files...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            generated = prepare_image_packets(
                image_path,
                os.path.abspath(output_dir),
                self.width_box.value(),
                self.height_box.value(),
                self.packet_len_box.value()
            )
        except (OSError, ValueError) as error:
            self.generated_files = {}
            self.generated_config = None
            self.refresh_selected_bin_label()
            self.show_error(f"BIN generation failed: {error}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.generate_btn.setEnabled(True)

        self.generated_files = generated
        self.generated_config = self.current_generation_config()
        self.refresh_selected_bin_label()

        self.show_image(self.tx_image, generated["reference"])
        self.tx_info.setText(
            f"Size: {self.width_box.value()} x {self.height_box.value()}\n"
            "BER: -\nPSNR: -"
        )

        messages = [
            "Image packet preparation complete",
            f"Reference: {generated['reference']}",
            f"Packet Length: {self.packet_len_box.value()}",
            f"COMB packets: {generated['combo_packets']}",
            f"COMB BIN: {generated['combo_whitened']}"
        ]
        if generated["normal_whitened"]:
            messages.extend([
                f"RM/OFDM packets: {generated['normal_packets']}",
                f"RM/OFDM BIN: {generated['normal_whitened']}"
            ])
        else:
            messages.append(generated["normal_error"])

        self.log_box.setPlainText("\n".join(messages))

        selected = self.selected_bin_path()
        if selected:
            self.status.setStyleSheet("")
            self.status.setText(
                f"BIN files generated for {self.width_box.value()} x "
                f"{self.height_box.value()} image"
            )
        else:
            self.show_error(
                generated["normal_error"]
                or "No BIN was generated for the selected mode."
            )

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    def update_mode_view(self, mode):
        show_rm = mode in ("RM", "COMB")
        show_ofdm = mode in ("OFDM", "COMB")

        self.rm_path_label.setVisible(show_rm)
        self.rm_path_row.setVisible(show_rm)
        self.ofdm_path_label.setVisible(show_ofdm)
        self.ofdm_path_row.setVisible(show_ofdm)

        self.rm_result.setVisible(show_rm)
        self.ofdm_result.setVisible(show_ofdm)
        self.refresh_selected_bin_label()

    def required_output_paths(self):
        mode = self.mode.currentText()
        if mode == "RM":
            return {"RM": self.rm_output_path.text().strip()}
        if mode == "OFDM":
            return {"OFDM": self.ofdm_output_path.text().strip()}
        return {
            "RM": self.rm_output_path.text().strip(),
            "OFDM": self.ofdm_output_path.text().strip()
        }

    # ------------------------------------------------------------------
    # GNU Radio process control
    # ------------------------------------------------------------------
    def start_transmission(self):
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.show_error("A GNU Radio process is already running.")
            return

        gr_file = os.path.abspath(self.gr_file_path.text().strip())
        selected_bin = self.selected_bin_path()
        bin_file = os.path.abspath(selected_bin) if selected_bin else ""
        generated_reference = self.generated_files.get("reference")
        reference_file = (
            os.path.abspath(generated_reference)
            if generated_reference
            else ""
        )
        requested_outputs = self.required_output_paths()
        outputs = {
            branch: os.path.abspath(path)
            for branch, path in requested_outputs.items()
            if path
        }

        error = self.validate_inputs(
            gr_file,
            bin_file,
            reference_file,
            requested_outputs
        )
        if error:
            self.show_error(error)
            return

        for output_path in outputs.values():
            output_dir = os.path.dirname(output_path)
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as create_error:
                self.show_error(
                    f"Cannot create result directory '{output_dir}': "
                    f"{create_error}"
                )
                return

        self.output_snapshot = {
            path: self.file_signature(path)
            for path in outputs.values()
        }
        self.current_outputs = outputs
        self.run_completed = False
        self.process_tail = ""
        self.log_box.clear()

        process_environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment_values = {
            "RM_OFDM_MODE": {
                "RM": "RM",
                "OFDM": "OFDM",
                "COMB": "comb"
            }[self.mode.currentText()],
            "RM_OFDM_BIN": bin_file,
            "RM_OFDM_REFERENCE_IMAGE": reference_file,
            "RM_OFDM_IMAGE_WIDTH": str(self.width_box.value()),
            "RM_OFDM_IMAGE_HEIGHT": str(self.height_box.value()),
            "RM_OFDM_PACKET_LEN": str(self.packet_len_box.value()),
            "RM_OFDM_RM_OUTPUT": os.path.abspath(
                self.rm_output_path.text().strip()
            ),
            "RM_OFDM_OFDM_OUTPUT": os.path.abspath(
                self.ofdm_output_path.text().strip()
            ),
            "PYTHONUNBUFFERED": "1"
        }

        output_directories = {
            os.path.dirname(path) for path in outputs.values()
        }
        if len(output_directories) == 1:
            environment_values["RM_OFDM_OUTPUT_DIR"] = output_directories.pop()

        for name, value in environment_values.items():
            process_environment.insert(name, value)

        self.process = QtCore.QProcess(self)
        self.process.setProcessEnvironment(process_environment)
        self.process.setWorkingDirectory(os.path.dirname(gr_file))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.errorOccurred.connect(self.handle_process_error)
        self.process.finished.connect(self.handle_process_finished)

        self.start_btn.setEnabled(False)
        self.status.setStyleSheet("")
        self.status.setText(
            f"Running {self.mode.currentText()} transmission..."
        )

        self.process.start(sys.executable, ["-u", gr_file])
        self.poll_timer.start()
        self.timeout_timer.start(self.RESULT_TIMEOUT_MS)

    def validate_inputs(
        self,
        gr_file,
        bin_file,
        reference_file,
        requested_outputs
    ):
        if not self.gr_file_path.text().strip():
            return "Select a GNU Radio Python file."
        if not os.path.isfile(gr_file):
            return f"GNU Radio Python file does not exist: {gr_file}"

        if self.generated_config != self.current_generation_config():
            return "Generate BIN files for the current image settings first."
        if not bin_file:
            reason = self.generated_files.get("normal_error")
            return reason or "No generated BIN is available for this mode."
        if not os.path.isfile(bin_file):
            return f"Generated BIN file does not exist: {bin_file}"
        bin_size = os.path.getsize(bin_file)
        if bin_size == 0 or bin_size % self.packet_len_box.value() != 0:
            return (
                "Generated BIN size is not an integer multiple of the "
                "configured Packet Length. Generate it again."
            )

        if not reference_file:
            return "Generate BIN files before starting the receiver."
        if not os.path.isfile(reference_file):
            return f"Generated reference image does not exist: {reference_file}"

        missing_outputs = [
            branch for branch, path in requested_outputs.items() if not path
        ]
        if missing_outputs:
            return (
                "Set result file path(s) for: " + ", ".join(missing_outputs)
            )

        try:
            with Image.open(reference_file) as image:
                image.verify()
        except (OSError, ValueError) as image_error:
            return f"Reference image is invalid: {image_error}"

        return None

    def read_process_output(self):
        if not self.process:
            return

        text = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if not text:
            return

        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.process_tail = (self.process_tail + text)[-4000:]

    def check_result_files(self):
        if self.run_completed:
            return

        for path in self.current_outputs.values():
            if not self.result_file_is_ready(path):
                return

        self.finish_run_successfully()

    def result_file_is_ready(self, path):
        signature = self.file_signature(path)
        if signature is None:
            return False

        if signature == self.output_snapshot.get(path):
            return False

        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            return False

        return True

    @staticmethod
    def file_signature(path):
        try:
            stat_result = os.stat(path)
        except OSError:
            return None
        return stat_result.st_mtime_ns, stat_result.st_size

    def finish_run_successfully(self):
        self.run_completed = True
        self.poll_timer.stop()
        self.timeout_timer.stop()

        try:
            self.update_results()
        except (OSError, ValueError) as error:
            self.finish_run_with_error(f"Cannot display result: {error}")
            return

        self.status.setStyleSheet("")
        self.status.setText(f"{self.mode.currentText()} transmission finished")
        self.start_btn.setEnabled(True)
        self.stop_process()

    def finish_run_with_error(self, message):
        self.run_completed = True
        self.poll_timer.stop()
        self.timeout_timer.stop()
        self.start_btn.setEnabled(True)
        self.status.setText(message)
        self.status.setStyleSheet("color: #b00020;")
        self.stop_process()

    def handle_timeout(self):
        missing = [
            branch
            for branch, path in self.current_outputs.items()
            if not self.result_file_is_ready(path)
        ]
        self.finish_run_with_error(
            "Timed out waiting for result file(s): " + ", ".join(missing)
        )

    def handle_process_error(self, process_error):
        del process_error
        if self.run_completed:
            return
        self.finish_run_with_error(
            f"GNU Radio process error: {self.process.errorString()}"
        )

    def handle_process_finished(self, exit_code, exit_status):
        del exit_status
        if self.run_completed:
            return

        if all(
            self.result_file_is_ready(path)
            for path in self.current_outputs.values()
        ):
            self.finish_run_successfully()
            return

        detail = self.process_tail.strip().splitlines()
        suffix = f" Last output: {detail[-1]}" if detail else ""
        self.finish_run_with_error(
            "GNU Radio exited before producing all result files "
            f"(exit code {exit_code}).{suffix}"
        )

    def stop_process(self):
        if not self.process:
            return
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()
            QtCore.QTimer.singleShot(3000, self.kill_process_if_running)

    def kill_process_if_running(self):
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill()

    # ------------------------------------------------------------------
    # Result display and metrics
    # ------------------------------------------------------------------
    def update_results(self):
        reference_path = os.path.abspath(self.generated_files["reference"])
        expected_size = (
            self.width_box.value(),
            self.height_box.value()
        )

        self.show_image(self.tx_image, reference_path)
        with Image.open(reference_path) as reference_image:
            reference_rgb = reference_image.convert("RGB")
            source_size = reference_rgb.size
            if reference_rgb.size != expected_size:
                reference_rgb = reference_rgb.resize(
                    expected_size,
                    Image.Resampling.LANCZOS
                )

        self.tx_info.setText(
            f"Source: {source_size[0]} x {source_size[1]}\n"
            f"Expected: {expected_size[0]} x {expected_size[1]}\n"
            "BER: -\nPSNR: -"
        )

        reference_array = np.asarray(reference_rgb, dtype=np.uint8)

        if "RM" in self.current_outputs:
            self.update_branch_result(
                self.current_outputs["RM"],
                self.rm_image,
                self.rm_info,
                reference_array,
                expected_size
            )

        if "OFDM" in self.current_outputs:
            self.update_branch_result(
                self.current_outputs["OFDM"],
                self.ofdm_image,
                self.ofdm_info,
                reference_array,
                expected_size
            )

    def update_branch_result(
        self,
        path,
        image_widget,
        info_widget,
        reference_array,
        expected_size
    ):
        self.show_image(image_widget, path)

        with Image.open(path) as received_image:
            received_rgb = received_image.convert("RGB")
            actual_size = received_rgb.size
            received_array = np.asarray(received_rgb, dtype=np.uint8)

        if actual_size != expected_size:
            info_widget.setText(
                f"Size: {actual_size[0]} x {actual_size[1]}\n"
                f"Expected: {expected_size[0]} x {expected_size[1]}\n"
                "BER: unavailable\nPSNR: unavailable"
            )
            return

        xor_bytes = np.bitwise_xor(reference_array, received_array)
        bit_errors = int(np.unpackbits(xor_bytes.reshape(-1)).sum())
        total_bits = reference_array.size * 8
        ber = bit_errors / total_bits if total_bits else 0.0

        difference = (
            reference_array.astype(np.float64)
            - received_array.astype(np.float64)
        )
        mse = float(np.mean(difference ** 2))
        psnr = (
            float("inf")
            if mse == 0
            else 10 * np.log10((255 ** 2) / mse)
        )
        psnr_text = "inf" if np.isinf(psnr) else f"{psnr:.2f}"

        info_widget.setText(
            f"Size: {actual_size[0]} x {actual_size[1]}\n"
            f"BER: {ber:.6f}\n"
            f"PSNR: {psnr_text} dB"
        )

    def show_image(self, widget, path):
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            raise ValueError(f"Invalid image file: {path}")
        self.show_pixmap(widget, pixmap)

    def show_pil_image(self, widget, image):
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        image_bytes = rgb_image.tobytes("raw", "RGB")
        qimage = QtGui.QImage(
            image_bytes,
            width,
            height,
            width * 3,
            QtGui.QImage.Format_RGB888
        ).copy()
        self.show_pixmap(widget, QtGui.QPixmap.fromImage(qimage))

    @staticmethod
    def show_pixmap(widget, pixmap):
        widget.setText("")
        widget.setPixmap(
            pixmap.scaled(
                widget.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )

    def show_error(self, message):
        self.status.setText(message)
        self.status.setStyleSheet("color: #b00020;")

    def closeEvent(self, event):
        self.poll_timer.stop()
        self.timeout_timer.stop()
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
                self.process.waitForFinished(1000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RadioImageUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
