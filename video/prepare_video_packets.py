#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import cv2
import numpy as np


PACKET_LEN = 274
HALF = PACKET_LEN // 2          # 137
HEADER_SIZE = 10
PAYLOAD_SIZE = HALF - HEADER_SIZE   # 127

WIDTH = 128
HEIGHT = 128

JPEG_QUALITY = 70
WHITEN_SEED_BASE = 2026

# First dummy packet is the normal warmup.
START_WARMUP = 1

# These packets let GNU Radio flush the final real packets
# through the OFDM/OAMP pipeline before EOF.
END_FLUSH = 32


def make_header(frame_id, packet_id, total_packets, frame_size):

    return np.array([
        (frame_id >> 8) & 0xff,
        frame_id & 0xff,

        (packet_id >> 8) & 0xff,
        packet_id & 0xff,

        (total_packets >> 8) & 0xff,
        total_packets & 0xff,

        (frame_size >> 24) & 0xff,
        (frame_size >> 16) & 0xff,
        (frame_size >> 8) & 0xff,
        frame_size & 0xff,
    ], dtype=np.uint8)


def whitening_mask(frame_id, packet_id):

    # IMPORTANT:
    # whitening is unique for every packet of every frame.
    seed = (
        WHITEN_SEED_BASE
        + frame_id * 65536
        + packet_id
    )

    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        256,
        size=PAYLOAD_SIZE,
        dtype=np.uint8
    )


def make_video_packet(frame_id, packet_id,
                      total_packets, frame_size,
                      payload):

    padded = np.zeros(
        PAYLOAD_SIZE,
        dtype=np.uint8
    )

    padded[:len(payload)] = payload

    mask = whitening_mask(
        frame_id,
        packet_id
    )

    whitened = padded ^ mask

    header = make_header(
        frame_id,
        packet_id,
        total_packets,
        frame_size
    )

    # One 137-byte branch.
    branch = np.concatenate([
        header,
        whitened
    ])

    assert len(branch) == HALF

    # Same source packet is supplied to RM and OFDM.
    combo = np.concatenate([
        branch,
        branch
    ])

    assert len(combo) == PACKET_LEN

    return combo


def make_dummy():

    # VideoReceiver explicitly ignores frame_id == 65535.
    header = make_header(
        65535,
        0,
        0,
        0
    )

    payload = np.zeros(
        PAYLOAD_SIZE,
        dtype=np.uint8
    )

    branch = np.concatenate([
        header,
        payload
    ])

    combo = np.concatenate([
        branch,
        branch
    ])

    return combo


def main():

    if len(sys.argv) < 2:
        print(
            "Usage: prepare_video_packets.py "
            "<input_video> [output_dir]"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path("video_test")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open input video: "
            + str(input_path)
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    source_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    packets = []

    # Start warmup.
    for _ in range(START_WARMUP):
        packets.append(
            make_dummy()
        )

    frame_id = 0
    real_packet_count = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                JPEG_QUALITY
            ]
        )

        if not ok:
            raise RuntimeError(
                f"JPEG encode failed "
                f"for frame {frame_id}"
            )

        jpeg = np.asarray(
            encoded,
            dtype=np.uint8
        ).reshape(-1)

        frame_size = len(jpeg)

        total_packets = math.ceil(
            frame_size / PAYLOAD_SIZE
        )

        print(
            f"[FRAME {frame_id}] "
            f"bytes={frame_size} "
            f"packets={total_packets}",
            flush=True
        )

        for packet_id in range(
            total_packets
        ):

            start = (
                packet_id
                * PAYLOAD_SIZE
            )

            end = min(
                start + PAYLOAD_SIZE,
                frame_size
            )

            payload = jpeg[
                start:end
            ]

            combo = make_video_packet(
                frame_id,
                packet_id,
                total_packets,
                frame_size,
                payload
            )

            packets.append(
                combo
            )

            real_packet_count += 1

        frame_id += 1

    cap.release()

    # Tail packets only flush the GNU Radio pipeline.
    for _ in range(END_FLUSH):
        packets.append(
            make_dummy()
        )

    if frame_id == 0:
        raise RuntimeError(
            "Input video has no readable frames"
        )

    output = np.concatenate(
        packets
    ).astype(
        np.uint8
    )

    path = (
        output_dir
        / "video_packets_combo_whitened.bin"
    )

    output.tofile(
        path
    )

    print()
    print(
        "================================="
    )
    print(
        "VIDEO PACKETS READY"
    )
    print(
        "================================="
    )
    print(
        "Source frames:",
        source_count
    )
    print(
        "Encoded frames:",
        frame_id
    )
    print(
        "Real packets:",
        real_packet_count
    )
    print(
        "Start warmup:",
        START_WARMUP
    )
    print(
        "End flush:",
        END_FLUSH
    )
    print(
        "Total combo packets:",
        len(packets)
    )
    print(
        "Output bytes:",
        len(output)
    )
    print(
        "Output:",
        path
    )


if __name__ == "__main__":
    main()
