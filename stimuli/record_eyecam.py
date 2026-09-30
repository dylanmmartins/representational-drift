# -*- coding: utf-8 -*-
"""
Record the eye camera (it's a thorlabs DCC1645C-HQ) to greyscale mp4. this is
triggered on an arduino loaded with firmata that acts as a DAQ and recieves
an input from the stimulus computer. so, a parallel signal to the one that
starts the microscope. use a t-junction or something to split a single
BNC output of that stimulus output arduino.

to run this one, conda env needs:

    pip install pylablib pyfirmata2 imageio-ffmpeg opencv-python numpy

then in anaconda prompt, run the script as:

    python record_eyecam.py

you can use the flag --no-trigger to skip triggering and start acquisition
right away. you can use --preview-only to not save anything (i.e., for when
you are aligning the camera and getting it into focus). This saves an mp4
video, a csv of timetamps, and a json of the acquision setttings. they are
automatically written with a date/timestamp in the name and should never
overwrite previous saves.

DMM, Sept 2026
"""

import argparse
import csv
import json
import os
import queue
import signal
import sys
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np
from pyfirmata2 import Arduino
from pylablib.devices import uc480

SAVEDIR = os.path.dirname(os.path.abspath(__file__))

class ManualTrigger:

    def __init__(self):
        self.time = None # time.perf_counter() of the trigger
        self.received = threading.Event()
        self.armed = threading.Event() # edges are ignored until the camera is streaming
        self.n_ignored = 0 # rising edges that arrived before arming
        self.level = False

    def arm(self):
        self.armed.set()

    def set_now(self):
        self.time = time.perf_counter()
        self.received.set()

    def close(self):
        pass


class Trigger(ManualTrigger):

    def __init__(self, port, pin):
        super().__init__()

        self.level = None

        print(f"Connecting to Arduino ({port if port != Arduino.AUTODETECT else 'autodetect'})...")

        self.board = Arduino(port) # blocks ~5 s while the Uno resets
        self.board.samplingOn()
        self.pin = self.board.get_pin(f'd:{pin}:i')
        self.pin.register_callback(self._on_change)
        self.pin.enable_reporting()

    def _on_change(self, value):

        t = time.perf_counter()

        if value and self.level is False and not self.received.is_set():
            if self.armed.is_set():
                self.time = t
                self.received.set()
            else:
                self.n_ignored += 1
        self.level = value

    def close(self):

        self.board.exit()


def setup_camera(args):

    cam = uc480.UC480Camera(cam_id=args.cam_id)
    cam.set_color_mode('mono8')

    if args.aoi is not None:
        x, y, w, h = args.aoi
        cam.set_roi(x, x + w, y, y + h, args.bin, args.bin)
    else:
        cam.set_roi(hbin=args.bin, vbin=args.bin)

    cam.set_pixel_rate()
    cam.set_frame_period(1. / args.fps)
    cam.set_exposure(min(args.exposure * 1e-3, cam.get_frame_period()))

    if args.gain is not None:
        cam.set_gains(master=args.gain)
    cam.set_frameskip_behavior('skip')

    return cam


def describe_camera(cam):
    """ Print the camera's actual settings and return them.
    """
    info = cam.get_device_info()
    roi = cam.get_roi()
    fps = 1. / cam.get_frame_period()
    exposure_ms = cam.get_exposure() * 1e3
    pixel_clock_mhz = cam.get_pixel_rate() / 1e6
    print(f'  {info.model} (serial {info.serial_number})')
    print(f'  AOI x={roi[0]}-{roi[1]}, y={roi[2]}-{roi[3]}, bin {roi[4]}x{roi[5]}, '
          f'pixel clock {pixel_clock_mhz:.0f} MHz')
    print(f'  Frame rate {fps:.2f} Hz, exposure {exposure_ms:.2f} ms')
    return info, roi, fps, exposure_ms, pixel_clock_mhz


def preview_only(args):
    # preview without writing anything to disk so you can get camera lined up and in focus, etc.

    aoi = args.aoi
    args.aoi = None

    print('Setting up camera (preview only, nothing is saved)...')
    cam = setup_camera(args)
    describe_camera(cam)
    if aoi is not None:
        print(f'  Green box: --aoi {" ".join(str(v) for v in aoi)}')
    print('\n>>> Previewing. Press q or Esc in the preview window (or Ctrl+C in terminal) to quit.')

    cv2.namedWindow('eyecam', cv2.WINDOW_NORMAL)
    cam.setup_acquisition(nframes=args.buffer)
    cam.start_acquisition()

    try:
        while True:
            try:
                cam.wait_for_frame(timeout=0.5)
            except uc480.uc480TimeoutError:
                continue
            frame = cam.read_newest_image()
            if frame is None:
                continue

            focus = cv2.Laplacian(frame, cv2.CV_64F).var()
            saturated = 100. * np.mean(frame >= 255)

            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if aoi is not None:
                x, y, w, h = aoi
                cv2.rectangle(img, (x // args.bin, y // args.bin),
                              ((x + w) // args.bin, (y + h) // args.bin), (0, 255, 0), 2)
            cv2.putText(img, f'PREVIEW (not recording)  focus {focus:.0f}  saturated {saturated:.1f}%',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow('eyecam', img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty('eyecam', cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop_acquisition()
        cam.close()
        cv2.destroyAllWindows()


def acquire(cam, trigger, frame_queue, state, stop, buffer_frames):

    try:
        cam.setup_acquisition(nframes=buffer_frames)
        cam.start_acquisition()
        
        while not stop.is_set():
            
            try:
                cam.wait_for_frame(timeout=0.5)
            except uc480.uc480TimeoutError:
                continue
            
            frames, infos = cam.read_multiple_images(return_info=True)
            t_read = time.perf_counter()
            
            if not frames:
                continue
            
            cam_t_last = infos[-1].timestamp_dev * 1e-7
            
            for frame, info in zip(frames, infos):
                
                cam_t = info.timestamp_dev * 1e-7
                host_t = t_read - (cam_t_last - cam_t)
                if trigger.received.is_set() and host_t >= trigger.time:
                    frame_queue.put((frame, info.framestamp, cam_t, host_t - trigger.time))
                    state['n_queued'] += 1

            state['latest'] = frames[-1]

    except Exception as exc:
        state['error'] = exc
        stop.set()

    finally:
        cam.stop_acquisition()


def write(frame_queue, paths, fps, crf, state):

    video = None
    csvfile = None
    prev_stamp = None

    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break
            frame, stamp, cam_t, host_t = item

            if video is None:

                h, w = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
                
                video = imageio_ffmpeg.write_frames(
                    paths['video'], (w, h), pix_fmt_in='gray', fps=fps, codec='libx264',
                    quality=None, macro_block_size=1,
                    output_params=['-crf', str(crf), '-preset', 'veryfast']
                )
                
                video.send(None)
                csvfile = open(paths['timestamps'], 'w', newline='')
                writer = csv.writer(csvfile)

                writer.writerow(['frame', 'cam_frame_id', 'cam_time_s', 'host_time_s'])
                state['frame_size'] = (w, h)

            video.send(np.ascontiguousarray(frame[:h, :w]))
            writer.writerow([state['n_written'], stamp, f'{cam_t:.7f}', f'{host_t:.6f}'])
            
            if prev_stamp is not None:
                state['n_dropped'] += max(0, stamp - prev_stamp - 1)
            prev_stamp = stamp
            state['n_written'] += 1
            
            if state['n_written'] % 600 == 0:
                csvfile.flush()
    finally:
        if video is not None:
            video.close()
        if csvfile is not None:
            csvfile.close()


def wait_for_stop(stop):
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            line = ''
        if line.strip().lower() == 'stop':
            stop.set()
        elif line == '':
            time.sleep(0.1)
        elif line.strip():
            print("Type 'stop' and press Enter to end the recording.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=str, default='COM3') #for the ARDUINO not the camera!
    parser.add_argument('--pin', type=int, default=2) # pin on the arduino that acts as the input
    parser.add_argument('--no-trigger', action='store_true', default=False)
    parser.add_argument('--cam-id', type=int, default=0) # 0 just means first avaliable camera
    parser.add_argument('--aoi', type=int, nargs=4, default=None, metavar=('X', 'Y', 'W', 'H')) # default is full sensor size
    parser.add_argument('--bin', type=int, default=1)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--exposure', type=float, default=10.) # in msec
    parser.add_argument('--gain', type=float, default=None)
    # H.264 quality: 0 is lossless, 17 visually lossless, higher is smaller and worse
    # there is no reason i'd every change this, but it's convenient to have it bundled with other args as a function input
    parser.add_argument('--crf', type=int, default=17) 
    parser.add_argument('--buffer', type=int, default=200) # buffer sz in frames
    parser.add_argument('--outdir', type=str, default=SAVEDIR)
    parser.add_argument('--no-preview', action='store_true', default=False)
    parser.add_argument('--preview-only', action='store_true', default=False)
    args = parser.parse_args()

    if args.preview_only:
        preview_only(args)
        return

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(args.outdir, f'eyecam_{timestamp}')
    paths = {
        'video': base + '.mp4',
        'timestamps': base + '_timestamps.csv',
        'settings': base + '_settings.json',
    }
    os.makedirs(args.outdir, exist_ok=True)

    if args.no_trigger:
        trigger = ManualTrigger()
    else:
        trigger = Trigger(args.port, args.pin)
        print(f'Connected and waiting for rising edge on pin {args.pin}.')

    print('setting up camera...')
    try:
        cam = setup_camera(args)
    except Exception:
        trigger.close()
        raise
    info, roi, actual_fps, exposure_ms, pixel_clock_mhz = describe_camera(cam)
    if actual_fps < args.fps - 0.5:
        print(f'{args.fps:.1f} Hz was requested but {actual_fps:.2f} Hz is the maximum for the current area of interest')

    if trigger.level is None:
        print('no report from the trigger pin yet... check the arduino')
    elif trigger.level:
        print('the trigger line is already high. recording starts on the next low->high edge.')

    state = {'latest': None, 'n_queued': 0, 'n_written': 0, 'n_dropped': 0,
             'frame_size': None, 'error': None}
    frame_queue = queue.Queue()
    stop = threading.Event()

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    acq_thread = threading.Thread(target=acquire, daemon=True,
                                  args=(cam, trigger, frame_queue, state, stop, args.buffer))
    write_thread = threading.Thread(target=write, daemon=True,
                                    args=(frame_queue, paths, actual_fps, args.crf, state))
    stop_thread = threading.Thread(target=wait_for_stop, args=(stop,), daemon=True)
    acq_thread.start()
    write_thread.start()
    stop_thread.start()

    while state['latest'] is None and not stop.is_set():
        time.sleep(0.01)
    if trigger.n_ignored:
        print(f'ignored {trigger.n_ignored} trigger pulses that arrived before the camera was ready')
    trigger.arm()

    if args.no_trigger:
        trigger.set_now()
        print('\n>>>Recording (used no trigger). Type `stop` and press Enter to end.')
    else:
        print('\n>>>Eyecamera is ready and waiting for trigger from stim computer')

    if not args.no_preview:
        cv2.namedWindow('eyecam', cv2.WINDOW_NORMAL)

    announced = False
    last_status = time.perf_counter()
    trigger_wallclock = None
    while not stop.is_set():
        if trigger.received.is_set() and not announced:
            trigger_wallclock = time.time() - (time.perf_counter() - trigger.time)
            if not args.no_trigger:
                print(f'Trigger received at {time.strftime("%H:%M:%S", time.localtime(trigger_wallclock))}, recording.')
            announced = True

        if not args.no_preview and state['latest'] is not None:
            img = cv2.cvtColor(state['latest'], cv2.COLOR_GRAY2BGR)
            if trigger.received.is_set():
                text, color = f'rec  {state["n_queued"]} frames', (0, 0, 255)
            else:
                text, color = 'ready and waiting for trigger', (0, 200, 255)
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow('eyecam', img)
            cv2.waitKey(30)
        else:
            time.sleep(0.03)

        if announced and time.perf_counter() - last_status > 60.:
            print(f'  {state["n_written"]} frames written, {state["n_dropped"]} dropped, '
                  f'{frame_queue.qsize()} waiting to be encoded')
            last_status = time.perf_counter()

    print('Stopping...')
    acq_thread.join()
    frame_queue.put(None)
    write_thread.join()
    cam.close()
    trigger.close()
    if not args.no_preview:
        cv2.destroyAllWindows()

    if state['error'] is not None:
        print(f'error during acquisition: {state["error"]!r}')

    if state['n_written'] == 0:
        print('no frames were recorded (no trigger received?). Nothing saved.')
        return

    settings = {
        'model': info.model,
        'serial_number': info.serial_number,
        'roi': {'hstart': roi[0], 'hend': roi[1], 'vstart': roi[2], 'vend': roi[3],
                'hbin': roi[4], 'vbin': roi[5]},
        'saved_frame_size': state['frame_size'],
        'fps': actual_fps,
        'exposure_ms': exposure_ms,
        'pixel_clock_mhz': pixel_clock_mhz,
        'crf': args.crf,
        'trigger': None if args.no_trigger else {'port': args.port, 'pin': args.pin},
        'trigger_wallclock': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(trigger_wallclock))
                             if trigger_wallclock else None,
        'n_frames': state['n_written'],
        'n_dropped': state['n_dropped'],
        'error': repr(state['error']) if state['error'] is not None else None,
    }
    with open(paths['settings'], 'w') as f:
        json.dump(settings, f, indent=2)

    print(f'Saved {state["n_written"]} frames ({state["n_dropped"]} dropped) to {paths["video"]}')
    print(f'Timestamps: {paths["timestamps"]}')
    print(f'Settings:   {paths["settings"]}')


if __name__ == '__main__':

    main()
