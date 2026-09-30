#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "./prepare_and_run.sh /path/to/input.mp4"
    exit 1
fi

INPUT="$1"

cd /mnt/data_storage/summer6/china/video_help/video_from_photo

echo "================================"
echo "1. Preparing video packets"
echo "================================"

mkdir -p video_test

rm -f current_tx.bin
rm -f video_test/video_packets_combo_whitened.bin
rm -f video_test/rx_video_rm.mp4
rm -f video_test/rx_video_ofdm.mp4

python3 prepare_video_packets.py "$INPUT" video_test

echo
echo "================================"
echo "2. Creating current_tx.bin"
echo "================================"

if [ ! -s video_test/video_packets_combo_whitened.bin ]; then
    echo "ERROR: video_packets_combo_whitened.bin was not created"
    exit 1
fi

cp video_test/video_packets_combo_whitened.bin current_tx.bin

echo "current_tx.bin:"
ls -lh current_tx.bin

echo
echo "Packets:"
python3 - <<'PY'
from pathlib import Path

p = Path("current_tx.bin")
size = p.stat().st_size

print("bytes =", size)
print("packets =", size // 274)
print("remainder =", size % 274)
PY

echo
echo "================================"
echo "3. Running GNU Radio"
echo "================================"

PYTHONUNBUFFERED=1 python3 run_video_until_complete.py 2>&1 | tee video_run_full.log
