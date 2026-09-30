# -*- coding: utf-8 -*-
"""
fm2p/sparse_noise_mapping.py

Receptive field mapping from sparse noise stimuli.

Functions
---------
calc_sparse_noise_STAs
    Spatial STAs from sparse noise; writes sparse_noise_receptive_fields.h5.
calc_sparse_noise_STA_reliability
    Full and split-half STAs, streamed from disk; writes sparse_noise.h5.
calc_STA_correlation
    Split-half STA correlations and responsive flag; saves reliability_stats_v2.npy.
sparse_noise_mapping
    Dispatch to split STAs, reliability, or single-STA mapping.


DMM, March 2026
"""


import os
from utils.gui_funcs import select_file
from utils.files import read_h5, write_h5
from sparse_noise import compute_calcium_sta_spatial, compute_split_STAs
from sparse_noise_mp import compute_split_STAs_mp
import argparse
import numpy as np
import platform
from tqdm import tqdm

os_name = platform.system()


def default_stimpath():
    """ OS-specific path to the sparse noise stimulus .npy. """

    if os_name == 'Linux':
        return '/home/dylan/Fast2/repdrift/sparse_noise_sequence_v10.npy'
    elif os_name == 'Windows':
        return r'J:\sparse_noise\sparse_noise_sequence_v7.npy'


def calc_sparse_noise_STAs(preproc_path=None, stimpath=None):
    """ Spatial STAs from sparse noise; writes sparse_noise_receptive_fields.h5.

    Parameters
    ----------
    preproc_path : str or None
        Preproc .h5; opens a picker if None.
    stimpath : str or None
        Stimulus .npy; OS-specific default if None.

    Returns
    -------
    dict_out : dict
    """

    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        if os_name == "Linux":
            stimpath = '/home/dylan/Fast2/sparse_noise/sparse_noise_sequence_v7.npy'
        elif os_name == "Windows":
            stimpath = r'J:\sparse_noise\sparse_noise_sequence_v7.npy'

    stimulus = np.load(stimpath)[:,:,:,0]

    data = read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(norm_spikes, 0)

    sta_all, lag_axis, delay = compute_calcium_sta_spatial(
        stimulus,
        norm_spikes,
        stimT,
        twopT,
        window=15,
        delay=np.zeros(n_cells)
    )

    dict_out = {
        'STAs': sta_all,
        'lag_axis': lag_axis,
        'delay': delay
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_receptive_fields.h5')
    write_h5(savepath, dict_out)

    return dict_out


def calc_sparse_noise_STA_reliability(preproc_path=None, stimpath=None,
                                       n_processes=None, window=13):
    """ Full and split-half STAs, streamed from disk; writes sparse_noise.h5.

    Parameters
    ----------
    preproc_path : str or None
        Preproc .h5; opens a picker if None.
    stimpath : str or None
        Stimulus .npy; OS-specific default if None.
    n_processes : int or None
        Worker count.
    window : int
        STA lag window in frames.
    """

    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = default_stimpath()

    print('  Loading preprocessed data.')
    data = read_h5(preproc_path)

    # Do NOT load the full stimulus into RAM -- the mp function streams it.
    spikes = data['s2p_spks']
    stimT  = data['stimT']
    twopT  = data['twopT']

    STA, STA1, STA2, r, lags = compute_split_STAs_mp(
        stimpath,
        spikes,
        stimT,
        twopT,
        window=window,
        n_processes=n_processes,
    )

    dict_out = {
        'STA':   STA,
        'STA1':  STA1,
        'STA2':  STA2,
        'lags':  lags,
        'jcorr': r,
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise.h5')
    print('  Writing {}...'.format(savepath))
    write_h5(savepath, dict_out)



def sparse_noise_mapping(prepath=None):

    if prepath is None:
        prepath = select_file(
            'Select sparse noise HDF file.',
            [('HDF','.h5'),]
        )

    calc_sparse_noise_STA_reliability(
        prepath
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--path', type=str, default=None)
    args = parser.parse_args()

    sparse_noise_mapping(args.path)
