# -*- coding: utf-8 -*-
"""
Present natural movie stimulus. Run in Psychopy's own
python.exe terminal with:

    python show_sparse_noise.py

DMM, Sept 2026
"""

from psychopy import visual, core, event
import numpy as np
from pyfirmata2 import Arduino
import csv
import os
import time

def main():

    num_frames = 4000
    max_dots = 6
    diameter_range = (15, 350) # bwtween 1 and 40 visual degrees
    on_time = 0.500 # 3 Hz
    num_repeats = 1
    shuffle = False # shuffling after generation breaks the black<->white transition check
    seed = 0
    stimulus_file = 'C:/Users/Goard Lab/Desktop/DMM/sparse_noise_timestamp_files/sparse_noise_sequence_v11.npy'
    timestamp_dir = 'C:/Users/Goard Lab/Desktop/DMM/sparse_noise_timestamp_files'
    use_trigger = True
    trigger_pin = 8
    pulse_duration = 0.005
    arduino_port = 'COM9'

    timestamp_file = os.path.join(timestamp_dir, 'sparse_noise_timestamps_{}.csv'.format(
        time.strftime('%Y%m%d_%H%M%S')))


    def non_overlapping_pos(existing_dots, new_diameter, max_attempts=1000):
        """ Generate a random (x, y) position that doesn't overlap with existing dots.
        """
        new_radius = new_diameter / 2
        for _ in range(max_attempts):
            pos_x = np.random.uniform(-monitor_x + new_radius, monitor_x - new_radius)
            pos_y = np.random.uniform(-monitor_y + new_radius, monitor_y - new_radius)

            overlap = False
            for dot in existing_dots:
                old_x, old_y = dot['pos']
                old_r = dot['diameter'] / 2
                dist = np.hypot(pos_x - old_x, pos_y - old_y)
                if dist < (new_radius + old_r):
                    overlap = True
                    break
            if not overlap:
                return pos_x, pos_y

        return pos_x, pos_y


    def check_illegal_transitions(prev_dots, curr_dots, tol=1e-6):
        """
        Check if any dots transition directly from black→white or white→black
        between frames (illegal transitions).

        prev_dots and curr_dots are lists of dicts like
            {'pos': (x, y), 'diameter': d, 'color': plus or minus 1}
        """
        illegal = False

        # Compare all dots that occupy roughly the same position
        for prev_dot in prev_dots:
            px, py = prev_dot['pos']
            pr = prev_dot['diameter'] / 2
            pcol = prev_dot['color']

            for curr_dot in curr_dots:
                cx, cy = curr_dot['pos']
                cr = curr_dot['diameter'] / 2
                ccol = curr_dot['color']

                # If they overlap, check if color flips illegally
                dist = np.hypot(px - cx, py - cy)
                if dist < (pr + cr):  # overlapping dots (same region)
                    if np.abs(pcol - ccol) > (2 - tol):  # black/white flip
                        illegal = True
                        break
            if illegal:
                break

        return illegal


    def generate_frame(max_dots, diameter_range):
        """ Generate one sparse noise frame... list of dot dictionaries.
        """
        n_dots = np.random.randint(1, max_dots + 1)
        frame_dots = []

        for _ in range(n_dots):
            diameter = np.random.uniform(*diameter_range)
            color = np.random.choice([1, -1])
            pos_x, pos_y = non_overlapping_pos(frame_dots, diameter)
            frame_dots.append({
                'diameter': diameter,
                'color': color,
                'pos': (pos_x, pos_y)
            })
        return frame_dots


    def legal_sparse_frames(num_frames, max_dots, diameter_range,
                            monitor_x, monitor_y, shuffle=False,
                            max_attempts=1000):
        """ Generate a list of sparse noise stimulus instructions,
        ensuring that no frame transitions directly from black<->white.
        """

        stim_instructions = []

        # first frame can be anything cause you're starting from grey
        prev_frame = generate_frame(max_dots, diameter_range)
        stim_instructions.append(prev_frame)

        # after that run checks
        for _ in range(1, num_frames):
            for attempt in range(max_attempts):
                curr_frame = generate_frame(max_dots, diameter_range)
                if not check_illegal_transitions(prev_frame, curr_frame):
                    stim_instructions.append(curr_frame)
                    prev_frame = curr_frame
                    break
            else:
                raise RuntimeError(f"Failed to generate legal frame after {max_attempts} attempts.")

        if shuffle:
            np.random.shuffle(stim_instructions)

        return stim_instructions


    def send_trigger_pulse(pin, duration):
        pin.write(1)
        core.wait(duration)
        pin.write(0)


    np.random.seed(seed)

    if use_trigger:

        print(f"Connecting to Arduino ({arduino_port if arduino_port != Arduino.AUTODETECT else 'autodetect'})...")
        board = Arduino(arduino_port)
        trigger_out = board.get_pin(f'd:{trigger_pin}:o')
        trigger_out.write(0)
        print(f'Connected. Trigger on pin {trigger_pin}, {pulse_duration * 1000:.1f} ms pulse.')

    win = visual.Window(
        color=[0, 0, 0],
        units='pix',
        fullscr=True,
        checkTiming=False,
        screen=1
    )

    monitor_x, monitor_y = win.size[0] // 2, win.size[1] // 2
    print(f'Monitor resolution: {win.size[0]}x{win.size[1]}')

    measured_fps = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
    if not measured_fps:
        print('Couldnt measure frame rate so falling back to nominal rate.')
        measured_fps = win.monitorFramePeriod and (1. / win.monitorFramePeriod)
        print('Using {:.2f} Hz'.format(measured_fps))

    if not measured_fps:
        win.close()
        board.exit()
        raise RuntimeError('Couldnt find any monitor rate, so aborting. Might be a problem with psychopy install?')


    frame_period = 1. / measured_fps
    print(f'Monitor: {measured_fps:.2f} Hz')

    generate = not os.path.exists(stimulus_file)

    if generate:
        print(f'{stimulus_file} does not exist... generating a new stimulus sequence (which will take a few minutes).')

        # genreate stim instructions
        stim_instructions = legal_sparse_frames(
            num_frames = num_frames,
            max_dots = max_dots,
            diameter_range = diameter_range,
            monitor_x = monitor_x,
            monitor_y = monitor_y,
            shuffle = shuffle
        )

        # then switch to pre-rendered frame 
        pre_rendered_frames = []

        for i, frame_dots in enumerate(stim_instructions):

            stims = [
                visual.Circle(
                    win,
                    radius=dot['diameter'] / 2,
                    pos=dot['pos'],
                    fillColor=[dot['color']] * 3,
                    lineColor=[dot['color']] * 3,
                    units='pix'
                )
                for dot in frame_dots
            ]

            pre_rendered_frames.append(stims)

        n_stim = len(pre_rendered_frames)
        print(f"Pre-rendered {n_stim} frames.")

        # throw it in a temp file for now
        partial_file = stimulus_file[:-len('.npy')] + '_partial.npy'
        recorded_frames = None

        def draw_frame(i):
            for dot in pre_rendered_frames[i]:
                dot.draw()

    else:
        print(f'Loading existing stimulus sequence from {stimulus_file}')
        saved_frames = np.load(stimulus_file, mmap_mode='r')
        n_stim = saved_frames.shape[0]
        frame_h, frame_w = saved_frames.shape[1:3]
        if (frame_w, frame_h) != tuple(win.size):
            print(f'THIS IS A MAJOR PROBLEM SO PROBABLY SHOULD CANCEL THE PRESENTATION: saved frames are {frame_w}x{frame_h} but the window is '
                f'{win.size[0]}x{win.size[1]}.... frames will be stretched to fill it.')
        print(f"Loaded {n_stim} frames.")

        img = visual.ImageStim(win, size=tuple(win.size), units='pix', flipVert=True)

        def to_contrast(frame):
            """ uint8 [0,255] -> float32 [-1,1]. Frames are greyscale, so keep one channel.
            """
            if frame.ndim == 3:
                frame = frame[:, :, 0]
            return frame.astype(np.float32) / 127.5 - 1.0

        def draw_frame(i):
            img.draw()

        img.image = to_contrast(saved_frames[0])


    frame_data = []
    completed = False

    def end_last_frame(offset):
        if frame_data and frame_data[-1]['offset_s'] is None:
            frame_data[-1]['offset_s'] = offset
            frame_data[-1]['duration_actual'] = offset - frame_data[-1]['onset_s']

    win.flip()
    print('\n>>> Ready. Start microscope with it set to wait for input trigger. Then press SPACE '
        'while inside of stimulus window on this computer. Hit ESC to abort.')
    start_keys = event.waitKeys(keyList=['space', 'escape'])

    history_clock = core.Clock()

    try:
        if 'escape' in start_keys:
            raise KeyboardInterrupt

        win.callOnFlip(history_clock.reset)
        if use_trigger:
            win.callOnFlip(send_trigger_pulse, trigger_out, pulse_duration)

        for rep in range(num_repeats):
            for i in range(n_stim):
                draw_frame(i)

                win.flip()
                onset_time = history_clock.getTime()
                end_last_frame(onset_time)
                frame_data.append({
                    'rep': rep,
                    'frame_index': i,
                    'onset_s': onset_time,
                    'offset_s': None,
                    'duration_actual': None,
                })

                if generate and rep == 0:
                    # front buffer now holds the dots
                    frame = np.asarray(win.getMovieFrame(buffer='front'))
                    win.movieFrames = []
                    if recorded_frames is None:
                        recorded_frames = np.lib.format.open_memmap(
                            partial_file, mode='w+', dtype=frame.dtype, shape=(n_stim,) + frame.shape)
                    recorded_frames[i] = frame

                elif not generate:
                    # upload the next frame's texture while this one is on screen
                    img.image = to_contrast(saved_frames[(i + 1) % n_stim])

                print(f'Frame {i}: onset={onset_time:.3f}')

                # Hold until half a refresh
                while history_clock.getTime() < onset_time + on_time - frame_period / 2:
                    if event.getKeys(['escape']):
                        raise KeyboardInterrupt

                    core.wait(0.001)

        completed = True

    except KeyboardInterrupt:
        print('Escape pressed, aborting.')

    finally:
        # grey screen
        win.flip()
        end_last_frame(history_clock.getTime())

        if use_trigger:
            trigger_out.write(0)
            board.exit()

        with open(timestamp_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['rep', 'frame_index', 'onset_s', 'offset_s',
                                                        'duration_actual'])
            writer.writeheader()
            writer.writerows(frame_data)
        print(f'Wrote {len(frame_data)} frame timestamps to {timestamp_file}')

        if generate and recorded_frames is not None:
            recorded_frames.flush()
            del recorded_frames
            if completed:
                os.replace(partial_file, stimulus_file)
                print(f'Saved {n_stim} frames to {stimulus_file}')
            else:
                print(f'!!! full sequence was not shown. if youre not expecting this message, go back to stim file and see how much was cut off')

        win.close()

    core.quit()


if __name__ == '__main__':
    
    main()