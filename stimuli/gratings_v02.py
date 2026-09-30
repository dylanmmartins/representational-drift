from psychopy import visual, core, event, monitors, logging
import numpy as np
from pyfirmata2 import Arduino
import csv
import os
import time

SCREEN_WIDTH_CM = 16.5
SCREEN_HEIGHT_CM = 12.5
VIEW_DIST_CM = 6.
SCREEN_PIX = (1360,768)
SCREEN_INDEX = 1
FULLSCREEN = True
GAMMA = 1.

N_ORI = 12
N_REPS = 10
SF = 0.05
TF = 2.
DUR = 3. # sec

TRIGGER_PIN = 8 # Arduino digital pin wired to the microscope trigger
PULSE_DURATION = 0.005 # seconds the trigger line is held HIGH
ARDUINO_PORT = 'COM9' # or Arduino.AUTODETECT

print('Full duration for {} repeats: {:.2f} min'.format(N_REPS, N_REPS * N_ORI * DUR / 60.))

SAVEDIR = os.path.dirname(os.path.abspath(__file__))


def send_trigger_pulse(pin, duration):
    pin.write(1)
    core.wait(duration)
    pin.write(0)


mon = monitors.Monitor('mouseMon')
mon.setWidth(SCREEN_WIDTH_CM)
mon.setDistance(VIEW_DIST_CM)
mon.setSizePix(list(SCREEN_PIX))
mon.setGamma(GAMMA)
mon.saveMon()

win = visual.Window(
    size=list(SCREEN_PIX),
    monitor='mouseMon',
    units='deg',
    fullscr=FULLSCREEN,
    screen=SCREEN_INDEX,
    color=[0.,0.,0.],
    colorSpace='rgb',
    allowGUI=False,
    waitBlanking=True
)

measured_fps = win.getActualFrameRate(
    nIdentical=20,
    nMaxFrames=100,
    nWarmUpFrames=20,
    threshold=1
)
if measured_fps is None:
    print('Could not measure frame rate reliably, falling back to reported nominal rate.')
    measured_fps = win.monitorFramePeriod and (1. / win.monitorFramePeriod)
if not measured_fps:
    win.close()
    raise RuntimeError('Could not determine monitor frame rate; aborting.')
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

orientations = np.linspace(0,360, N_ORI, endpoint=False)

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

# one row per trial: [trial, orientation_deg, sf_cpd, tf_hz, repeat, onset_s, offset_s, n_frames]
# onset/offset are seconds since the microscope trigger pulse
log_rows = []

def end_last_trial(offset):
    if log_rows and log_rows[-1][6] is None:
        log_rows[-1][6] = offset

# Connect to Arduino and send a TTL pulse at stimulus onset
print(f"Connecting to Arduino ({ARDUINO_PORT if ARDUINO_PORT != Arduino.AUTODETECT else 'autodetect'})...")
board = Arduino(ARDUINO_PORT)
trigger_out = board.get_pin(f'd:{TRIGGER_PIN}:o')
trigger_out.write(1)
print(f'  Connected. Trigger on pin {TRIGGER_PIN}, {PULSE_DURATION * 1000:.1f} ms pulse.')

print('Sending start trigger to microscope...')
send_trigger_pulse(trigger_out, PULSE_DURATION)

history_clock = core.MonotonicClock()

try:
    for _ in range(int(2. * FRAME_RATE)): # 2 sec grey screen at start
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

            # the first flip of this trial is its onset and the previous trial's offset
            if i_frame == 0:
                onset = history_clock.getTime()
                end_last_trial(onset)
                log_rows.append([i_trial, ori, SF, TF, rep, onset, None, n_frames])

            if event.getKeys(keyList=['escape']):
                log_rows[-1][7] = i_frame + 1
                raise KeyboardInterrupt

    # the first grey flip after the last trial is that trial's offset
    win.flip()
    end_last_trial(history_clock.getTime())
    for _ in range(int(2.*FRAME_RATE) - 1): # 2 sec grey screen at end
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
    print(f'Wrote {len(log_rows)} trials to {log_fname}')

    win.close()

core.quit()
