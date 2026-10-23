# -*- coding: utf-8 -*-
"""
analyze.py

DMM, October 2026
"""

import os
import sys
import glob
import argparse
import traceback

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from files import find
from gui_funcs import select_directory
import analyze_gratings
import analyze_movie
import analyze_sparse_noise
from sparse_noise_mapping import default_stimpath
from activity import SOURCES, DEFAULT_SOURCE
from summary_doc import make_summary_doc

MOVIES = {
    'antelopes': 'antelope',
    'meerkats': 'meerkat',
    'penguins': 'penguin',
}
STIMULI = ['gratings'] + list(MOVIES) + ['sparse_noise']


def find_recording_dirs(session_dir):

    rec_dirs = {}
    for stim in STIMULI:
        path = os.path.join(session_dir, stim)
        if os.path.isdir(path):
            rec_dirs[stim] = path

    return rec_dirs


def find_timestamps(rec_dir, pattern):

    path = find(pattern, rec_dir, MR=True, retempty=True)
    if path is None:
        matches = glob.glob(os.path.join(os.path.dirname(rec_dir), pattern))
        if len(matches) == 0:
            raise FileNotFoundError('No {} in {} or its parent directory'.format(pattern, rec_dir))
        path = max(matches, key=os.path.getmtime)

    return path


def find_stimpath(session_dir):

    matches = glob.glob(os.path.join(session_dir, 'sparse_noise_sequence*.npy'))
    if len(matches) > 0:
        return max(matches, key=os.path.getmtime)
    return default_stimpath()


def main(session_dir, skip_existing=False, n_shuffles=1000, activity_source=DEFAULT_SOURCE,
         stimpath=None):

    session_dir = os.path.abspath(session_dir)
    rec_dirs = find_recording_dirs(session_dir)
    print('Session {}'.format(session_dir))
    print('Activity: {}'.format(activity_source))
    for stim in STIMULI:
        print('  {:<13} {}'.format(stim, rec_dirs.get(stim, 'not found')))
    if len(rec_dirs) == 0:
        raise FileNotFoundError('No recording directories named {} in {}'.format(STIMULI, session_dir))

    outputs = {}
    failed = {}

    def run(stim, out_path, analysis):

        if skip_existing and os.path.isfile(out_path):
            print('\n=== {}: using existing {}'.format(stim, out_path))
            outputs[stim] = out_path
            return
        print('\n=== {}'.format(stim))
        try:
            analysis()
            outputs[stim] = out_path
        except Exception:
            traceback.print_exc()
            failed[stim] = rec_dirs[stim]

    if 'gratings' in rec_dirs:
        gt_dir = rec_dirs['gratings']
        run('gratings', os.path.join(gt_dir, 'gratings.h5'), lambda: analyze_gratings.main(
            gt_dir,
            time_path=find_timestamps(gt_dir, 'gratings_timestamps*.csv'),
            n_shuffles=n_shuffles,
            activity_source=activity_source))

    for movie, prefix in MOVIES.items():
        if movie in rec_dirs:
            mv_dir = rec_dirs[movie]
            run(movie, os.path.join(mv_dir, 'movie.h5'), lambda: analyze_movie.main(
                mv_dir,
                time_path=find_timestamps(mv_dir, '{}_timestamps*.csv'.format(prefix)),
                n_shuffles=n_shuffles,
                activity_source=activity_source))

    if 'sparse_noise' in rec_dirs:
        sn_dir = rec_dirs['sparse_noise']
        sn_cfg = dict(analyze_sparse_noise.cfg,
                      stimpath=stimpath if stimpath is not None else find_stimpath(session_dir),
                      activity_source=activity_source)
        print('Sparse noise stimulus: {}'.format(sn_cfg['stimpath']))
        run('sparse_noise', os.path.join(sn_dir, 'sparse_noise.h5'),
            lambda: analyze_sparse_noise.main(sn_dir, cfg=sn_cfg))

    if failed:
        print('\nFailed (left out of the summary): {}'.format(', '.join(failed)))

    print('\n=== summary PDF')
    make_summary_doc(
        os.path.join(session_dir, 'cell_summary.pdf'),
        gratings_path=outputs.get('gratings'),
        movie_paths=[outputs[m] for m in MOVIES if m in outputs],
        sparse_noise_path=outputs.get('sparse_noise'),
        title=os.path.basename(session_dir),
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, help='Session directory')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Reuse gratings.h5 / movie.h5 / sparse_noise.h5 already in a recording directory')
    parser.add_argument('--n_shuffles', type=int, default=1000)
    parser.add_argument('--activity', choices=SOURCES, default=DEFAULT_SOURCE,
                        help='omsi_prob: OMSI spike probability; omsi_train: OMSI spike train; s2p: suite2p spks')
    parser.add_argument('--stimpath', default=None,
                        help='Sparse noise stimulus .npy (default: sparse_noise_sequence*.npy in the session directory)')
    args = parser.parse_args()

    if args.dir is None:
        session_dir = select_directory('Select session directory')
    else:
        session_dir = args.dir

    main(session_dir, skip_existing=args.skip_existing, n_shuffles=args.n_shuffles,
         activity_source=args.activity, stimpath=args.stimpath)

# python analyze.py --path /your/path/to/the/directory