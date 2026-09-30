"""Packet receiver for the existing 10-byte video protocol.
Reference bytes are used for measurement only, never to repair received data.
This simulation benchmark assumes the transmitted reference is available locally.
"""
import json
from pathlib import Path
import cv2
import numpy as np


def header(packet):
    return (int.from_bytes(bytes(packet[:2]), 'big'),
            int.from_bytes(bytes(packet[2:4]), 'big'),
            int.from_bytes(bytes(packet[4:6]), 'big'),
            int.from_bytes(bytes(packet[6:10]), 'big'))


def unwhite(payload, fid, pid, seed):
    rng = np.random.default_rng(seed + fid * 65536 + pid)
    return payload ^ rng.integers(0, 256, len(payload), dtype=np.uint8)


class VideoReceiver:
    def __init__(self, reference, folder, width, height, seed):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.width, self.height, self.fps = width, height, 30.0
        meta_path = self.folder / 'video_meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.width, self.height = int(meta['width']), int(meta['height'])
            self.fps = float(meta['fps'])
        if self.width <= 0 or self.height <= 0 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Invalid video dimensions/FPS')
        self.refs, self.frames = {}, {}
        self.half = reference.shape[1] // 2
        self.capacity = self.half - 10
        if self.capacity <= 0:
            raise ValueError('Video packet too short')
        for full in reference:
            for branch, packet in zip(('RM', 'OFDM'), (full[:self.half], full[self.half:])):
                fid, pid, total, size = header(packet)
                if fid == 65535:
                    continue
                if total != (size + self.capacity - 1) // self.capacity or size <= 0 or not 0 <= pid < total:
                    raise ValueError('Invalid TX reference header; regenerate packets')
                key = (branch, fid, pid)
                if key in self.refs and not np.array_equal(self.refs[key], packet):
                    raise ValueError('Conflicting TX reference packet')
                self.refs[key] = packet.copy()
                framekey = (branch, fid)
                if framekey in self.frames and self.frames[framekey] != (total, size):
                    raise ValueError('Conflicting frame metadata')
                self.frames[framekey] = (total, size)
        if not self.refs:
            raise ValueError('No video frames in TX reference')
        for (branch, fid), (total, _) in self.frames.items():
            if any((branch, fid, pid) not in self.refs for pid in range(total)):
                raise ValueError('Incomplete TX reference')
        self.received = {}
        self.stats = {b: dict(errors=0, bits=0, invalid=0, duplicates=0) for b in ('RM', 'OFDM')}
        self.saved = False
        self.report = None

    @property
    def complete(self):
        return len(self.received) == len(self.refs)

    def accept(self, packet, branch):
        if self.saved:
            return 0, 0
        stats = self.stats[branch]
        if len(packet) != self.half:
            stats['invalid'] += 1
            return 0, 0
        fid, pid, total, size = header(packet)

        if fid >= 40:
            print(
                "VIDEO HEADER:",
                branch,
                "fid=", fid,
                "pid=", pid,
                "total=", total,
                "size=", size,
                flush=True
            )

        if fid == 65535:
            return 0, 0
        key = (branch, fid, pid)
        expected = self.refs.get(key)

        if expected is not None:
            print(
                "HEADER DEBUG:",
                branch,
                "RX:",
                list(packet[:10]),
                "TX:",
                list(expected[:10])
            )

        # Header validity/alignment is separate from payload BER.
        if expected is None or not np.array_equal(packet[:10], expected[:10]):
            stats['invalid'] += 1

            if stats['invalid'] <= 20:
                print(
                    "HEADER FAIL:",
                    branch,
                    "RX=",
                    list(packet[:10]),
                    "TX=",
                    list(expected[:10]) if expected is not None else None
                )

            return 0, 0
        if key in self.received:
            stats['duplicates'] += 1
            return 0, 0
        real = min(self.capacity, size - pid * self.capacity)
        payload = packet[10:].copy()
        errors = int(np.unpackbits(payload[:real] ^ expected[10:10 + real]).sum())
        bits = real * 8
        if fid >= 40:
            print(
                "ACCEPT PACKET:",
                branch,
                "fid=",
                fid,
                "pid=",
                pid,
                flush=True
            )

        self.received[key] = payload
        stats['errors'] += errors
        stats['bits'] += bits
        return errors, bits

    def save(self):
        if self.saved:
            return self.report
        report = {'fps': self.fps, 'width': self.width, 'height': self.height,
                  'policy': 'first received packet; no reference-based repair; incomplete frames omitted',
                  'ber_scope': 'unique packets with valid reference-aligned headers, real payload only',
                  'branches': {}}
        for branch in ('RM', 'OFDM'):
            path = self.folder / ('rx_video_' + branch.lower() + '.mp4')
            # This directory belongs to this run; do not leave an old video looking successful.
            if path.exists():
                path.unlink()
            writer = None
            decoded, exact, incomplete, failed = [], [], [], []
            try:
                for (b, fid), (total, size) in sorted(self.frames.items()):
                    if b != branch:
                        continue
                    keys = [(b, fid, pid) for pid in range(total)]
                    if any(k not in self.received for k in keys):
                        incomplete.append(fid)
                        continue
                    data = np.concatenate([unwhite(self.received[k], fid, k[2], self.seed) for k in keys])[:size]
                    expected = np.concatenate([unwhite(self.refs[k][10:], fid, k[2], self.seed) for k in keys])[:size]
                    if np.array_equal(data, expected):
                        exact.append(fid)
                    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[:2] != (self.height, self.width):
                        failed.append(fid)
                        continue
                    if writer is None:
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                                 (self.width, self.height))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open video writer: ' + str(path))
                    writer.write(frame)
                    decoded.append(fid)
            finally:
                if writer is not None:
                    writer.release()
            stats = dict(self.stats[branch])
            stats.update(ber=stats['errors'] / stats['bits'] if stats['bits'] else None,
                         expected_packets=sum(k[0] == branch for k in self.refs),
                         received_packets=sum(k[0] == branch for k in self.received),
                         expected_frames=sum(k[0] == branch for k in self.frames),
                         decoded_frames=decoded, exact_jpeg_frames=exact,
                         incomplete_frames=incomplete, decode_failed_frames=failed,
                         output=str(path) if decoded else None)
            report['branches'][branch] = stats
            print(f"[VIDEO_RX] {branch}: decoded={len(decoded)}/{stats['expected_frames']} "
                  f"exact={len(exact)} BER={stats['ber']} incomplete={incomplete}", flush=True)
        (self.folder / 'video_rx_report.json').write_text(json.dumps(report, indent=2))
        self.report, self.saved = report, True
        return report
