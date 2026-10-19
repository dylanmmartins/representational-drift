# -*- coding: utf-8 -*-
"""
representational_drift/utils/activity.py

Per-frame activity used by the stimulus analyses.

omsi_prob... OMSI posterior spike probability (prob_trace) x frame rate, in spikes/s
omsi_train...  OMSI spike train binned to frames x frame rate, in spikes/s
s2p...     suite2p (OASIS) spks.npy, arbitrary units

DMM, October 2026
"""

import os
import numpy as np

from twop import run_omsi, OMSI_ENV


SOURCES = ('omsi_prob', 'omsi_train', 's2p')
DEFAULT_SOURCE = 'omsi_prob'
LABELS = {
    'omsi_prob': 'OMSI spikes/s',
    'omsi_train': 'OMSI spikes/s',
    's2p': 'suite2p spks (a.u.)',
}
CACHE_NAME = 'omsi_spikes.npz'


def frame_rate(twopT):

    return float(1.0 / np.nanmedian(np.diff(twopT)))


def pad_spike_times(spike_times):

    out = np.full((len(spike_times), max(max((len(st) for st in spike_times), default=1), 1)), np.nan)
    for i, st in enumerate(spike_times):
        out[i, :len(st)] = st
    return out


def spike_times_to_train(spike_times, n_frames, fs):
    
    train = np.zeros((len(spike_times), n_frames), dtype=np.float32)
    for i, st in enumerate(spike_times):
        st = np.asarray(st, dtype=float)
        st = st[np.isfinite(st)]
        idx = np.clip(np.floor(st * fs + 1e-9).astype(int), 0, n_frames - 1)
        np.add.at(train[i], idx, 1)
    return train


def run_omsi_cached(plane_dir, F, Fneu, usecells, fs, neu_correction=0.7, env=OMSI_ENV):

    cache_path = os.path.join(plane_dir, CACHE_NAME)
    if os.path.isfile(cache_path):
        c = np.load(cache_path)
        if (np.array_equal(c['usecells'], usecells)
                and c['prob_trace'].shape[1] == F.shape[1]
                and np.isclose(c['hz'], fs)
                and np.isclose(c['neu_correction'], neu_correction)):
            print('  Using cached OMSI output {}'.format(cache_path))
            return c['prob_trace'], c['spike_times']
        print('  Cached OMSI output {} does not match this recording; rerunning.'.format(cache_path))

    print('  OMSI spike inference on {} cells at {:.3f} Hz...'.format(np.size(F, 0), fs))
    omsi = run_omsi(F, Fneu, fs, env=env, f_corr=neu_correction)
    if omsi['spike_times'] is None or omsi['prob_trace'] is None:
        raise RuntimeError('OMSI failed on {}'.format(plane_dir))

    prob_trace = np.asarray(omsi['prob_trace'], dtype=np.float32)
    spike_times = pad_spike_times(omsi['spike_times'])
    np.savez(cache_path, prob_trace=prob_trace, spike_times=spike_times,
             usecells=usecells, hz=fs, neu_correction=neu_correction)
    print('  Wrote {}'.format(cache_path))

    return prob_trace, spike_times


def load_activity(plane_dir, fs, source=DEFAULT_SOURCE, neu_correction=0.7, omsi_env=OMSI_ENV):

    if source not in SOURCES:
        raise ValueError('activity source must be one of {}, got {!r}'.format(SOURCES, source))

    iscell = np.load(os.path.join(plane_dir, 'iscell.npy'), allow_pickle=True)
    usecells = iscell[:, 0] == 1

    if source == 's2p':
        spks = np.load(os.path.join(plane_dir, 'spks.npy'), allow_pickle=True)
        return spks[usecells]

    F = np.load(os.path.join(plane_dir, 'F.npy'), allow_pickle=True)[usecells]
    Fneu = np.load(os.path.join(plane_dir, 'Fneu.npy'), allow_pickle=True)[usecells]

    prob_trace, spike_times = run_omsi_cached(
        plane_dir, F, Fneu, usecells, fs, neu_correction=neu_correction, env=omsi_env)

    if source == 'omsi_prob':
        return prob_trace * fs
    return spike_times_to_train(spike_times, F.shape[1], fs) * fs
