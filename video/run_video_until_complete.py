#!/usr/bin/env python3

import sys
import time

from PyQt5 import QtWidgets
from Ran_ofdm import Ran_ofdm


app = QtWidgets.QApplication.instance()

if app is None:
    app = QtWidgets.QApplication(
        sys.argv
    )


tb = Ran_ofdm()

receiver = (
    tb.customModule_ber_cal_0
      .video_receiver
)

total = len(
    receiver.refs
)

noise = float(
    getattr(
        tb,
        "noise_volt",
        0.0
    )
)


print(
    "[VIDEO_RUNNER]",
    "packets=",
    total,
    "noise=",
    noise,
    "FEC=RS40",
    flush=True
)


tb.start()

start = time.time()

last = -1
last_change = time.time()

MAX_SECONDS = 600.0


try:

    while True:

        app.processEvents()

        current = len(
            receiver.received
        )

        if current != last:

            last = current
            last_change = time.time()

            percent = (
                100.0
                * current
                / total
                if total
                else 0.0
            )

            print(
                f"[VIDEO_PROGRESS] "
                f"{current}/{total} "
                f"({percent:.1f}%)",
                flush=True
            )


        # FEC receiver only counts
        # successfully decoded packets.
        if receiver.complete:

            print(
                "[VIDEO_RUNNER] COMPLETE",
                flush=True
            )

            break


        idle = (
            time.time()
            - last_change
        )

        elapsed = (
            time.time()
            - start
        )


        if (
            noise > 0
            and elapsed > 30
            and idle > 12
        ):

            print(
                "[VIDEO_RUNNER] "
                "no more recoverable packets",
                flush=True
            )

            break


        if elapsed > MAX_SECONDS:

            print(
                "[VIDEO_RUNNER] TIMEOUT",
                flush=True
            )

            break


        time.sleep(
            0.05
        )


finally:

    if not receiver.saved:

        tb.customModule_ber_cal_0.save_video()

    tb.stop()
    tb.wait()


print(
    "[VIDEO_RUNNER] finished",
    flush=True
)
