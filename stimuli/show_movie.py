# -*- coding: utf-8 -*-
"""
Present natural movie stimulus. This can present three movies:
antelopes, penguins, and meerkats. Run in Psychopy's own
python.exe terminal. Run with:

    python show_movie.py --movie A

For the `--movie` flag, options are A, P, and M (for each 
animal video).

DMM, Sept 2026
"""

import argparse
import csv
import sys
import time
import os

import numpy as np
import tifffile
from psychopy import core, event, visual

from pyfirmata2 import Arduino


STIM_FPS = 30.0
TRIGGER_PIN = 8
PULSE_DURATION = 0.005
SERIAL_PORT = 'COM9'
SEED = 0 # this shouldnt matter or be used anywhere... just here out of caution


def to_contrast(frame):
    return frame.astype(np.float32) / 127.5 - 1.0

def send_trigger_pulse(pin, duration):
    pin.write(1)
    core.wait(duration)
    pin.write(0)

SAVEDIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", help='Which movie will be the target to present? e.g., A for antelopes.', default='P')
    parser.add_argument("--repeats", type=int, default=50, help="Number of times to loop the movie")
    parser.add_argument("--pin", type=int, default=TRIGGER_PIN, help="Arduino digital pin for the trigger pulse")
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION, help="Trigger pulse width (s)")
    parser.add_argument("--port", type=str, default=SERIAL_PORT, help="Arduino serial port (default: autodetect)")
    parser.add_argument("--screen", type=int, default=1, help="Screen index for the psychopy window")
    parser.add_argument("--windowed", action="store_true", default=False, help="Run in a window instead of fullscreen")
    parser.add_argument("--log", type=str, default=None, help="CSV path to log per-frame flip timestamps")
    parser.add_argument("--startF", type=int, default=-1, help="Frame index to start playback from")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--no-wait", action="store_true", default=False,
                        help="Start immediately instead of waiting for SPACE after arming the microscope")
    args = parser.parse_args()

    np.random.seed(args.seed)

    startF = args.startF
    if args.movie == 'M':
        stimstr = 'meerkat'
        if args.startF == -1:
            startF = 165
    elif args.movie == 'P':
        stimstr = 'penguin'
        if args.startF == -1:
            startF = 0
    elif args.movie == 'A':
        stimstr = 'antelope'
        if args.startF == -1:
            startF = 0
    else:
        raise ValueError('movie argument was not one of {M, A, P}')

    if args.log is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.log = os.path.join(SAVEDIR, f'{stimstr}_timestamps_{timestamp}.csv')

    
    if args.movie == 'M':
        movie_path = 'C:/Users/Goard Lab/Desktop/DMM/repdrift_stim/meerkats.tif'
    elif args.movie == 'P':
        movie_path = 'C:/Users/Goard Lab/Desktop/DMM/repdrift_stim/penguins.tif'
    elif args.movie == 'A':
        movie_path = 'C:/Users/Goard Lab/Desktop/DMM/repdrift_stim/antelopes.tif'

    print(f"Loading stimulus stack: {movie_path}")
    stack = tifffile.memmap(movie_path)
    print(stack.shape)
    stack = np.transpose(stack, (0,2,1)) # was 2, 0, 1
    stack = stack[startF:, :, :]
    n_frames, height, width = stack.shape
    print(f"  {n_frames} frames, {width}x{height}, {args.repeats} repeat")

    print(f"Connecting to Arduino ({args.port if args.port != Arduino.AUTODETECT else 'autodetect'})...")
    try:
        board = Arduino(args.port)
    except Exception as exc:
        sys.exit(f"Could not connect to Arduino: {exc}")
    trigger_pin = board.get_pin(f"d:{args.pin}:o")
    trigger_pin.write(0)
    print(f"Connected. Trigger on pin {args.pin}, {args.pulse_duration * 1000:.1f} msec pulse.")

    win = visual.Window(
        fullscr=not args.windowed,
        screen=args.screen,
        color="black",
        units="pix",
        allowGUI=False,
    )
    win.mouseVisible = False

    print("Measuring actual frame rate of monitor...")
    measured_fps = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
    if measured_fps is None:
        print('Couldnt measure frame rate so falling back to nominal rate.')
        measured_fps = win.monitorFramePeriod and (1.0 / win.monitorFramePeriod)
        print('Using {:.2f} Hz'.format(measured_fps))

    if not measured_fps:
        win.close()
        board.exit()
        raise RuntimeError('Couldnt find any monitor rate, so aborting. Might be a problem with psychopy install?')

    frames_per_stim_frame = max(1, round(measured_fps / STIM_FPS))
    print(f"Monitor is {measured_fps:.2f} Hz, and holding each stimulus frame for {frames_per_stim_frame} refreshes")
    print(f'Expected duration of complete presentation (one repeat): {n_frames * frames_per_stim_frame / measured_fps:.2f} s')
    print('each frame lasts {:.2f} ms'.format(1000 * frames_per_stim_frame / measured_fps))
    print('Full presentation of {} repeats will last {:.2f} min'.format(args.repeats, (args.repeats * n_frames * frames_per_stim_frame / measured_fps) / 60.))

    img = visual.ImageStim(win, size=(2.,2.), units="norm")
    img.flipVert = True

    log_rows = []

    img.image = to_contrast(stack[0])

    win.flip()
    start_keys = []
    if not args.no_wait:
        print('\n>>> Ready. Start microscope with it set to wait for input trigger. Then press SPACE '
            'while inside of stimulus window on this computer. Hit ESC to abort.')
        start_keys = event.waitKeys(keyList=["space", "escape"])

    history_clock = core.Clock()

    try:
        if "escape" in start_keys:
            raise KeyboardInterrupt

        win.callOnFlip(history_clock.reset)
        win.callOnFlip(send_trigger_pulse, trigger_pin, args.pulse_duration)
        for rep in range(args.repeats):
            for frame_idx in range(n_frames):
                if frame_idx > 0 or rep > 0:
                    img.image = to_contrast(stack[frame_idx])
                for i_refresh in range(frames_per_stim_frame):
                    img.draw()
                    win.flip()
                    if i_refresh == 0:
                        onset = history_clock.getTime()

                if log_rows:
                    log_rows[-1][3] = onset
                log_rows.append([rep, startF + frame_idx, onset, None])

                if event.getKeys(keyList=["escape"]):
                    print("Escape pressed, aborting playback.")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:

        win.flip()
        if log_rows:
            log_rows[-1][3] = history_clock.getTime()

        trigger_pin.write(0)
        board.exit()
        win.close()

        n_dropped = sum(1 for d in win.frameIntervals if d > (1.5 / measured_fps)) if win.frameIntervals else 0
        print(f"Done. {len(log_rows)} stimulus frames presented. Approx. dropped monitor frames: {n_dropped}")

        if args.log:
            with open(args.log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["repeat", "movie_frame", "onset_s", "offset_s"])
                writer.writerows(log_rows)
            print(f"Frame timing log written to {args.log}")


if __name__ == "__main__":
    
    main()
