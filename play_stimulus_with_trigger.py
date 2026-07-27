"""
Play a natural-movie stimulus (as saved by make_movie_stacks.ipynb) in PsychoPy
at its native 30 Hz frame rate, and fire a single TTL pulse on an Arduino
(running StandardFirmata) at stimulus onset to trigger microscope acquisition.

Requires:
    pip install psychopy pyfirmata2

Hardware:
    - Arduino loaded with the "StandardFirmata" sketch (File > Examples >
      Firmata > StandardFirmata in the Arduino IDE).
    - TRIGGER_PIN wired to the microscope's external-trigger input, with a
      shared ground between the Arduino and the microscope's trigger board.

Usage:
    python play_stimulus_with_trigger.py natural_movie_one.tif --repeats 10
"""

import argparse
import csv
import sys
import time

import numpy as np
import tifffile
from psychopy import core, event, visual

from pyfirmata2 import Arduino

# ----------------------------------------------------------------------
# Config (override via CLI flags below)
# ----------------------------------------------------------------------
STIM_FPS = 30.0            # native frame rate of the Allen natural-movie stimuli
TRIGGER_PIN = 8             # Arduino digital pin wired to the microscope trigger
PULSE_DURATION = 0.005       # seconds the trigger line is held HIGH
SERIAL_PORT = 'COM5' # Arduino.AUTODETECT  # or e.g. '/dev/ttyACM0' / 'COM3'


def to_contrast(frame):
    """uint8 [0,255] -> float32 [-1,1] for psychopy's default rgb colorSpace."""
    return frame.astype(np.float32) / 127.5 - 1.0


def send_trigger_pulse(pin, duration):
    pin.write(1)
    core.wait(duration)
    pin.write(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", help="Path to a .tif stimulus stack (n_frames, H, W)",
                            default='C:/Users/GoardLab/Desktop/repdrift/natural_movie_three.tif')
    parser.add_argument("--repeats", type=int, default=10, help="Number of times to loop the movie")
    parser.add_argument("--pin", type=int, default=TRIGGER_PIN, help="Arduino digital pin for the trigger pulse")
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION, help="Trigger pulse width (s)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT, help="Arduino serial port (default: autodetect)")
    parser.add_argument("--screen", type=int, default=1, help="Screen index for the psychopy window")
    parser.add_argument("--windowed", action="store_true", default=False, help="Run in a window instead of fullscreen")
    parser.add_argument("--log", type=str, default=None, help="CSV path to log per-frame flip timestamps")
    args = parser.parse_args()

    print(f"Loading stimulus stack: {args.movie}")
    stack = tifffile.memmap(args.movie)
    n_frames, height, width = stack.shape
    print(f"  {n_frames} frames, {width}x{height}, {args.repeats} repeat(s)")

    print(f"Connecting to Arduino ({args.port if args.port != Arduino.AUTODETECT else 'autodetect'})...")
    try:
        board = Arduino(args.port)
    except Exception as exc:
        sys.exit(f"Could not connect to Arduino: {exc}")
    trigger_pin = board.get_pin(f"d:{args.pin}:o")
    trigger_pin.write(0)
    print(f"  Connected. Trigger on pin {args.pin}, {args.pulse_duration * 1000:.1f} ms pulse.")

    win = visual.Window(
        fullscr=not args.windowed,
        screen=args.screen,
        color="black",
        units="pix",
        allowGUI=False,
    )
    win.mouseVisible = False

    print("Measuring actual monitor frame rate...")
    measured_fps = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
    if measured_fps is None:
        print("  Could not measure frame rate reliably, falling back to reported nominal rate.")
        measured_fps = win.monitorFramePeriod and (1.0 / win.monitorFramePeriod)
    if not measured_fps:
        board.exit()
        win.close()
        sys.exit("Could not determine monitor frame rate; aborting.")

    frames_per_stim_frame = max(1, round(measured_fps / STIM_FPS))
    print(f"  Monitor: {measured_fps:.2f} Hz -> holding each stimulus frame for {frames_per_stim_frame} refresh(es)")

    aspect = width / height
    img = visual.ImageStim(win, size=(aspect,2.), units="norm")
    img.flipVert = True

    log_rows = []

    try:
        img.image = to_contrast(stack[0])
        img.draw()

        print("Sending start trigger to microscope...")
        send_trigger_pulse(trigger_pin, args.pulse_duration)
        t_start = core.getTime()

        for rep in range(args.repeats):
            for frame_idx in range(n_frames):
                if frame_idx > 0 or rep > 0:
                    img.image = to_contrast(stack[frame_idx])
                for _ in range(frames_per_stim_frame):
                    img.draw()
                    flip_time = win.flip()
                log_rows.append((rep, frame_idx, flip_time - t_start))

                if event.getKeys(keyList=["escape"]):
                    print("Escape pressed, aborting playback.")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        trigger_pin.write(0)
        board.exit()
        win.close()

        n_dropped = sum(1 for d in win.frameIntervals if d > (1.5 / measured_fps)) if win.frameIntervals else 0
        print(f"Done. {len(log_rows)} stimulus frames presented. Approx. dropped monitor frames: {n_dropped}")

        if args.log:
            with open(args.log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["repeat", "frame_idx", "t_since_trigger_s"])
                writer.writerows(log_rows)
            print(f"Frame timing log written to {args.log}")


if __name__ == "__main__":
    main()
