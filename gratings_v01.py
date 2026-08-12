from psychopy import visual, core, event, monitors, logging
import numpy as np
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

print('Full duration for {} repeats: {:.2f} min'.format(N_REPS, N_REPS * DUR / 60.))

SAVEDIR = os.path.dirname(os.path.abspath(__file__))

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
if measured_fps is not None:
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
log_fname = os.path.join(SAVEDIR, f'drifting_gratings_{timestamp}.csv')
log_file = open(log_fname, 'w')
log_file.write('trial,orientation_deg,repeat,onset_time_s,offset_time_s,n_frames')

global_clock = core.Clock()
for _ in range(int(2. * FRAME_RATE)): # 2 sec grey screen at start
    win.flip()
    
for i_trial, trial in enumerate(trials):
    
    ori = trial['orientation']
    rep = trial['repeat']
    
    grating.ori = ori
    grating.phase=0.
    n_frames = int(round(DUR * FRAME_RATE))
    trial_clock = core.Clock()
    
    onset = global_clock.getTime()
    
    for i_frame in range(n_frames):
        
        grating.phase += PHASE_PER_FRAME
        grating.draw()
        win.flip()
        
        keys = event.getKeys(keyList=['escape'], timeStamped=False)
        
        if keys:
            offset = global_clock.getTime()
            log_file.write(f'{i_trial},{ori},{rep},{offset-onset:.6f},{onset:.6f},{offset:.6f},{i_frame+1}\n')
            log_file.close()
            win.close()
            core.quit()
            
        offset = global_clock.getTime()
        log_file.write(f'{i_trial},{ori},{rep},{onset:.6f},{offset:.6f},{n_frames}\n')
        
for _ in range(int(2.*FRAME_RATE)):
    win.flip()
    
log_file.close()
win.close()
core.quit()

    

    
    