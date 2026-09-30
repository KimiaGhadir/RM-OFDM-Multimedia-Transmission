import os
import numpy as np
from gnuradio import gr
from PIL import Image


class ber_cal(gr.sync_block):

    def __init__(
        self,
        seq_length,
        seq,
        mod_type,
        image_mode=False,
        image_width=128,
        image_height=128,
        image_channels=3,
        image_save_dir="/tmp",
        whitening_seed=2026,
        warmup_packets=1
    ):

        gr.sync_block.__init__(
            self,
            name="ber_cal",
            in_sig=[(np.uint8, seq_length)],
            out_sig=[np.float32, np.float32]
        )

        self.length = int(seq_length)
        self.seq = np.asarray(seq, dtype=np.uint8)
        self.mod_type = str(mod_type)

        self.combo_mode = self.mod_type in (
            "comb",
            "comb_alt",
            "comb_swap"
        )

        if self.combo_mode and self.length % 2 != 0:
            raise ValueError(
                "Combo mode requires an even seq_length."
            )

        self.branch_length = (
            self.length // 2
            if self.combo_mode
            else self.length
        )

        self.image_mode = bool(image_mode)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.image_channels = int(image_channels)
        self.image_save_dir = str(image_save_dir)
        self.whitening_seed = int(whitening_seed)
        self.warmup_packets = int(warmup_packets)

        # BER terminal logging
        self.ber_log_every = int(
            os.environ.get("BER_LOG_EVERY", "20")
        )
        self.ber_log_calls = 0

        self.OFDM_error_count = 0
        self.RM_error_count = 0
        self.total_bits = 0

        # Separate counters are required in combo mode.
        self.OFDM_total_bits = 0
        self.RM_total_bits = 0

        self.normal_packet_index = 0

        if len(self.seq) % self.length != 0:
            raise ValueError(
                "Sequence length must be an integer multiple of seq_length."
            )

        self.reference_packets = self.seq.reshape(
            -1,
            self.length
        )

        if self.image_mode:

            self.total_packets = len(self.reference_packets)

            # First byte is Packet ID.
            # Standalone image packet:
            # [1-byte ID][payload]
            #
            # Combo branch:
            # [2-byte ID][payload]
            self.image_id_bytes = (
                2 if self.combo_mode else 1
            )

            self.data_per_packet = (
                self.branch_length
                - self.image_id_bytes
            )

            self.image_size = (
                self.image_width
                * self.image_height
                * self.image_channels
            )

            # Number of actual image packets is determined by
            # image size, not by total seq length.
            self.num_packets = int(
                np.ceil(
                    self.image_size
                    / self.data_per_packet
                )
            )

            required_packets = (
                self.warmup_packets
                + self.num_packets
            )

            if self.total_packets < required_packets:
                raise ValueError(
                    "Reference sequence does not contain enough "
                    "packets for the configured image."
                )

            # Received image chunks indexed by Packet ID.
            # Standalone receiver state.
            self.rx_packets = {}
            self.image_saved = False

            # Combo receiver state.
            self.rx_packets_rm = {}
            self.rx_packets_ofdm = {}

            self.image_saved_rm = False
            self.image_saved_ofdm = False

            self.image_dir = self.image_save_dir

            os.makedirs(
                self.image_dir,
                exist_ok=True
            )

    def byte_to_bits(self, byte):

        return [
            (int(byte) >> i) & 1
            for i in range(7, -1, -1)
        ]

    def count_byte_errors(self, sent, received):

        sent_bits = self.byte_to_bits(sent)
        received_bits = self.byte_to_bits(received)

        return sum(
            a != b
            for a, b in zip(
                sent_bits,
                received_bits
            )
        )

    def handle_image_packet(self, packet):

        # BER and reconstruction refer to the first image cycle.
        if self.image_saved:
            return 0, 0

        packet_id = int(packet[0])
        
        # Ignore corrupted / invalid packet IDs.
        if (
            packet_id < 0
            or packet_id >= self.num_packets
        ):
            return 0, 0

        # Do not count the same image packet more than once.
        if packet_id in self.rx_packets:
            return 0, 0

        # Skip configured warm-up packets.
        start = (
            self.warmup_packets
            + packet_id
        ) * self.length

        expected_packet = self.seq[
            start:start + self.length
        ]

        # Packet ID is metadata. BER is calculated only
        # over REAL image payload bytes.
        # Padding in the final packet is excluded.
        image_start = (
            packet_id
            * self.data_per_packet
        )

        valid_payload_bytes = min(
            self.data_per_packet,
            self.image_size - image_start
        )

        expected_payload = expected_packet[
            1:1 + valid_payload_bytes
        ]

        received_payload = packet[
            1:1 + valid_payload_bytes
        ]

        errors = 0

        for sent, received in zip(
            expected_payload,
            received_payload
        ):
            errors += self.count_byte_errors(
                sent,
                received
            )

        bits = valid_payload_bytes * 8

        # De-whiten the received image payload.
        image_payload = np.asarray(
            packet[1:],
            dtype=np.uint8
        ).copy()

        rng = np.random.default_rng(
            self.whitening_seed + packet_id
        )

        mask = rng.integers(
            0,
            256,
            size=self.data_per_packet,
            dtype=np.uint8
        )

        image_payload ^= mask

        # Keep the first received version
        # of each packet.
        if packet_id not in self.rx_packets:
            self.rx_packets[packet_id] = image_payload
            
        # Reconstruct/update the image whenever we reach
        # the end of one transmission cycle.
        if (
            packet_id == self.num_packets - 1
            and not self.image_saved
        ):

            # Start with a black image buffer.
            image_bytes = np.zeros(
                self.image_size,
                dtype=np.uint8
            )

            # Insert every packet that has been received.
            for pid, payload in self.rx_packets.items():

                start = pid * self.data_per_packet
                end = min(
                    start + self.data_per_packet,
                    self.image_size
                )

                if start >= self.image_size:
                    continue

                image_bytes[start:end] = payload[
                    :end - start
                ]

            image_array = image_bytes.reshape(
                self.image_height,
                self.image_width,
                self.image_channels
            )

            if self.mod_type == "RM":
                filename = "rx_rm.png"

            elif self.mod_type == "OFDM":
                filename = "rx_ofdm.png"

            else:
                filename = "rx_image.png"

            output_path = os.path.join(
                self.image_dir,
                filename
            )

            Image.fromarray(
                image_array,
                mode="RGB"
            ).save(output_path)

            missing = [
                i
                for i in range(self.num_packets)
                if i not in self.rx_packets
            ]

            print(
                f"[IMAGE_RX] Saved: {output_path} "
                f"| received={len(self.rx_packets)}/"
                f"{self.num_packets} "
                f"| missing={missing}"
            )
            
            self.image_saved = True

        return errors, bits

    def handle_combo_image_branch(
        self,
        packet,
        branch
    ):

        # Combo uses a big-endian 2-byte Packet ID.
        packet_id = (
            (int(packet[0]) << 8)
            | int(packet[1])
        )

        if branch == "RM":
            rx_packets = self.rx_packets_rm
            already_saved = self.image_saved_rm

        else:
            rx_packets = self.rx_packets_ofdm
            already_saved = self.image_saved_ofdm

        # Keep the first completed image cycle.
        if already_saved:
            return 0, 0

        # Invalid Packet ID cannot be aligned reliably.
        if (
            packet_id < 0
            or packet_id >= self.num_packets
        ):
            return 0, 0

        # Do not count duplicate packets twice.
        if packet_id in rx_packets:
            return 0, 0

        reference_index = (
            self.warmup_packets
            + packet_id
        )

        expected_combo_packet = (
            self.reference_packets[
                reference_index
            ]
        )

        if branch == "RM":
            expected_packet = (
                expected_combo_packet[
                    :self.branch_length
                ]
            )
        else:
            expected_packet = (
                expected_combo_packet[
                    self.branch_length:
                ]
            )

        image_start = (
            packet_id
            * self.data_per_packet
        )

        valid_payload_bytes = min(
            self.data_per_packet,
            self.image_size - image_start
        )

        if valid_payload_bytes <= 0:
            return 0, 0

        expected_payload = expected_packet[
            2:2 + valid_payload_bytes
        ]

        received_payload = packet[
            2:2 + valid_payload_bytes
        ]

        errors = 0

        for sent, received in zip(
            expected_payload,
            received_payload
        ):
            errors += self.count_byte_errors(
                sent,
                received
            )

        bits = valid_payload_bytes * 8

        # Full payload is needed for reconstruction.
        image_payload = np.asarray(
            packet[2:],
            dtype=np.uint8
        ).copy()

        rng = np.random.default_rng(
            self.whitening_seed
            + packet_id
        )

        mask = rng.integers(
            0,
            256,
            size=self.data_per_packet,
            dtype=np.uint8
        )

        image_payload ^= mask

        rx_packets[packet_id] = image_payload

        if packet_id == self.num_packets - 1:
            self.save_combo_image(branch)

        return errors, bits


    def save_combo_image(self, branch):

        if branch == "RM":
            rx_packets = self.rx_packets_rm
            filename = "rx_rm.png"
        else:
            rx_packets = self.rx_packets_ofdm
            filename = "rx_ofdm.png"

        image_bytes = np.zeros(
            self.image_size,
            dtype=np.uint8
        )

        for pid, payload in rx_packets.items():

            start = (
                pid
                * self.data_per_packet
            )

            end = min(
                start + self.data_per_packet,
                self.image_size
            )

            if start >= self.image_size:
                continue

            image_bytes[start:end] = payload[
                :end - start
            ]

        if self.image_channels == 1:

            image_array = image_bytes.reshape(
                self.image_height,
                self.image_width
            )

            pil_mode = "L"

        else:

            image_array = image_bytes.reshape(
                self.image_height,
                self.image_width,
                self.image_channels
            )

            pil_mode = (
                "RGB"
                if self.image_channels == 3
                else "RGBA"
            )

        output_path = os.path.join(
            self.image_dir,
            filename
        )

        Image.fromarray(
            image_array,
            mode=pil_mode
        ).save(output_path)

        missing = [
            pid
            for pid in range(self.num_packets)
            if pid not in rx_packets
        ]

        print(
            f"[IMAGE_RX_{branch}] "
            f"Saved: {output_path} "
            f"| received={len(rx_packets)}/"
            f"{self.num_packets} "
            f"| missing={missing}"
        )

        if branch == "RM":
            self.image_saved_rm = True
        else:
            self.image_saved_ofdm = True


    def _maybe_print_ber(self, ofdm_ber, rm_ber):
        """Print the actual BER values written to output streams."""

        self.ber_log_calls += 1

        if self.ber_log_every <= 0:
            return

        if self.ber_log_calls % self.ber_log_every != 0:
            return

        print(
            "[BER_LOG] "
            f"calls={self.ber_log_calls} "
            f"OFDM_BER={float(ofdm_ber):.12g} "
            f"RM_BER={float(rm_ber):.12g}",
            flush=True
        )


    def work(self, input_items, output_items):

        in0 = input_items[0]

        out0 = output_items[0]
        out1 = output_items[1]

        for i in range(len(in0)):

            packet = in0[i]

            # -------------------------
            # COMBO IMAGE MODE
            # -------------------------
            if self.image_mode and self.combo_mode:

                rm_packet = packet[
                    :self.branch_length
                ]

                ofdm_packet = packet[
                    self.branch_length:
                ]

                rm_errors, rm_bits = (
                    self.handle_combo_image_branch(
                        rm_packet,
                        "RM"
                    )
                )

                ofdm_errors, ofdm_bits = (
                    self.handle_combo_image_branch(
                        ofdm_packet,
                        "OFDM"
                    )
                )

                if rm_bits > 0:
                    self.RM_error_count += rm_errors
                    self.RM_total_bits += rm_bits

                if ofdm_bits > 0:
                    self.OFDM_error_count += ofdm_errors
                    self.OFDM_total_bits += ofdm_bits

                RM_ber = (
                    self.RM_error_count
                    / self.RM_total_bits
                    if self.RM_total_bits > 0
                    else 0.0
                )

                OFDM_ber = (
                    self.OFDM_error_count
                    / self.OFDM_total_bits
                    if self.OFDM_total_bits > 0
                    else 0.0
                )

                # Output 0 = OFDM BER
                # Output 1 = RM BER
                out0[i] = OFDM_ber
                out1[i] = RM_ber

                continue

            # -------------------------
            # IMAGE TRANSMISSION MODE
            # -------------------------
            if self.image_mode:

                errors, bits = (
                    self.handle_image_packet(
                        packet
                    )
                )

                if bits == 0:

                    # No new valid bits were received.
                    # Keep displaying the cumulative BER
                    # instead of resetting the GUI to zero.
                    if self.total_bits > 0:

                        if self.mod_type == "OFDM":
                            out0[i] = (
                                self.OFDM_error_count
                                / self.total_bits
                            )
                            out1[i] = 0

                        elif self.mod_type == "RM":
                            out0[i] = 0
                            out1[i] = (
                                self.RM_error_count
                                / self.total_bits
                            )

                        else:
                            out0[i] = 0
                            out1[i] = 0

                    else:
                        out0[i] = 0
                        out1[i] = 0

                    continue

                if self.mod_type == "RM":

                    self.RM_error_count += errors
                    self.total_bits += bits

                    RM_ber = (
                        self.RM_error_count
                        / self.total_bits
                    )

                    out0[i] = 0
                    out1[i] = RM_ber

                elif self.mod_type == "OFDM":

                    self.OFDM_error_count += errors
                    self.total_bits += bits

                    OFDM_ber = (
                        self.OFDM_error_count
                        / self.total_bits
                    )

                    out0[i] = OFDM_ber
                    out1[i] = 0

                else:

                    out0[i] = 0
                    out1[i] = 0

                continue

            # -------------------------
            # ORIGINAL NORMAL MODE
            # -------------------------

            if self.mod_type == "OFDM":

                for j in range(self.length):

                    self.OFDM_error_count += (
                        self.count_byte_errors(
                            self.reference_packets[
                                self.normal_packet_index
                                % len(self.reference_packets),
                                j
                            ],
                            packet[j]
                        )
                    )

                self.total_bits += (
                    self.length * 8
                )

                self.normal_packet_index += 1

                out0[i] = (
                    self.OFDM_error_count
                    / self.total_bits
                )

                out1[i] = 0

            elif self.mod_type == "RM":

                for j in range(self.length):

                    self.RM_error_count += (
                        self.count_byte_errors(
                            self.reference_packets[
                                self.normal_packet_index
                                % len(self.reference_packets),
                                j
                            ],
                            packet[j]
                        )
                    )

                self.total_bits += (
                    self.length * 8
                )

                self.normal_packet_index += 1

                out0[i] = 0

                out1[i] = (
                    self.RM_error_count
                    / self.total_bits
                )

            else:

                out0[i] = 0
                out1[i] = 0

        if len(in0) > 0:
            self._maybe_print_ber(
                out0[len(in0) - 1],
                out1[len(in0) - 1]
            )

        return len(in0)
