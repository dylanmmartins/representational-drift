# -*- coding: utf-8 -*-
"""
Present gratings stimulus. You need to run this in
Psychopy's own python.exe terminal. Run with:

    python show_gratings.py

DMM, Sept 2026
"""

from psychopy import visual, core, event, monitors, logging
import numpy as np
from pyfirmata2 import Arduino
import csv
import os
import time

SCREEN_WIDTH_CM = 16.5
SCREEN_HEIGHT_CM = 12.5
VIEW_DIST_CM = 6.
SCREEN_INDEX = 1
FULLSCREEN = True
GAMMA = 1.

N_ORI = 12
N_REPS = 10
SF = 0.05
TF = 2.
DUR = 3. # in sec
SEED = 0 # for the trial order

TRIGGER_PIN = 8
PULSE_DURATION = 0.005 # also in sec
ARDUINO_PORT = 'COM9'

SAVEDIR = os.path.dirname(os.path.abspath(__file__))


def main():

    print('Full duration for {} repeats: {:.2f} min'.format(N_REPS, N_REPS * N_ORI * DUR / 60.))

    np.random.seed(SEED)

    def send_trigger_pulse(pin, duration):
        pin.write(1)
        core.wait(duration)
        pin.write(0)

    print(f"Connecting to Arduino ({ARDUINO_PORT if ARDUINO_PORT != Arduino.AUTODETECT else 'autodetect'})...")
    board = Arduino(ARDUINO_PORT)
    trigger_out = board.get_pin(f'd:{TRIGGER_PIN}:o')
    trigger_out.write(0)
    print(f'Connected. Trigger on pin {TRIGGER_PIN}, {PULSE_DURATION * 1000:.1f} msec pulse.')

    mon = monitors.Monitor('mouseMon')
    mon.setWidth(SCREEN_WIDTH_CM)
    mon.setDistance(VIEW_DIST_CM)
    mon.setGamma(GAMMA)

    win = visual.Window(
        monitor=mon,
        units='deg',
        fullscr=FULLSCREEN,
        screen=SCREEN_INDEX,
        color=[0.,0.,0.],
        colorSpace='rgb',
        allowGUI=False,
        waitBlanking=True
    )

    win.monitor.setSizePix(list(win.size))
    mon.setSizePix(list(win.size))
    mon.saveMon()
    print(f'Monitor resolution: {win.size[0]}x{win.size[1]}')

    measured_fps = win.getActualFrameRate(
        nIdentical=20,
        nMaxFrames=100,
        nWarmUpFrames=20,
        threshold=1
    )
    if measured_fps is None:
        print('Couldnt measure frame rate so falling back to nominal rate.')
        measured_fps = win.monitorFramePeriod and (1. / win.monitorFramePeriod)
        print('Using {:.2f} Hz'.format(measured_fps))

    if not measured_fps:
        win.close()
        board.exit()
        raise RuntimeError('Couldnt find any monitor rate, so aborting. Might be a problem with psychopy install?')

    FRAME_RATE = round(measured_fps)

    FRAME_DUR = 1. / FRAME_RATE
    PHASE_PER_FRAME = TF / FRAME_RATE

    GRATING_SIZE_DEG = (180., 180.)

    grating = visual.GratingStim(
        win,
        tex='sin',
        mask=None,
        units='deg',
        size=GRATING_SIZE_DEG,
        sf=SF,
        phase=0.,
        ori=0.,
        contrast=1.,
        interpolate=True,
        autoLog=False
    )

    orientations = np.linspace(0, 360, N_ORI, endpoint=False)

    trials = []
    for ori in orientations:
        for rep in range(N_REPS):
            trials.append({
                'orientation': ori,
                'repeat': rep,
                'duration': DUR
            })

    np.random.shuffle(trials)
    n_trails = len(trials)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_fname = os.path.join(SAVEDIR, f'gratings_timestamps_{timestamp}.csv')

    log_rows = []

    def end_last_trial(offset):
        if log_rows and log_rows[-1][6] is None:
            log_rows[-1][6] = offset

    win.flip()
    print('\n>>> Ready. Start microscope with it set to wait for input trigger. Then press SPACE '
        'while inside of stimulus window on this computer. Hit ESC to abort.')
    start_keys = event.waitKeys(keyList=['space', 'escape'])

    history_clock = core.Clock()

    try:
        if 'escape' in start_keys:
            raise KeyboardInterrupt

        win.callOnFlip(history_clock.reset)
        win.callOnFlip(send_trigger_pulse, trigger_out, PULSE_DURATION)
        for _ in range(int(2. * FRAME_RATE)): # 2 sec grey screen
            win.flip()

        for i_trial, trial in enumerate(trials):

            ori = trial['orientation']
            rep = trial['repeat']

            grating.ori = ori
            grating.phase=0.
            n_frames = int(round(DUR * FRAME_RATE))

            for i_frame in range(n_frames):

                grating.phase += PHASE_PER_FRAME
                grating.draw()
                win.flip()

                if i_frame == 0:
                    onset = history_clock.getTime()
                    end_last_trial(onset)
                    log_rows.append([i_trial, ori, SF, TF, rep, onset, None, n_frames])

                if event.getKeys(keyList=['escape']):
                    log_rows[-1][7] = i_frame + 1
                    raise KeyboardInterrupt

        win.flip()
        end_last_trial(history_clock.getTime())
        for _ in range(int(2.*FRAME_RATE) - 1):
            win.flip()

    except KeyboardInterrupt:
        print('Escape pressed, aborting.')
        win.flip()
        end_last_trial(history_clock.getTime())

    finally:
        trigger_out.write(0)
        board.exit()

        with open(log_fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'orientation_deg', 'sf_cpd', 'tf_hz', 'repeat',
                            'onset_s', 'offset_s', 'n_frames'])
            writer.writerows(log_rows)
        print(f'Wrote {len(log_rows)} trials to {log_fname} (seed={SEED})')

        win.close()

    core.quit()


if __name__ == "__main__":
    
    main()