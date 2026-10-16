# -*- coding: utf-8 -*-
"""
fm2p/sparse_noise_mapping.py

Receptive field mapping from sparse noise stimuli.


DMM, March 2026
"""


import os
from gui_funcs import select_file
from files import read_h5, write_h5
from sparse_noise import compute_calcium_sta_spatial, compute_split_STAs
from sparse_noise_fast import compute_split_STAs_fast
import argparse
import numpy as np
import platform
from tqdm import tqdm

os_name = platform.system()


def default_stimpath():

    if os_name == 'Linux':
        return '/home/dylan/Fast2/repdrift/sparse_noise_sequence_v10.npy'
    elif os_name == 'Windows':
        return r'J:\sparse_noise\sparse_noise_sequence_v7.npy'


def calc_sparse_noise_STAs(preproc_path=None, stimpath=None):

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
                                       window=13, downsample=4):

    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = default_stimpath()

    print('  Loading preprocessed data.')
    data = read_h5(preproc_path)

    spikes = data['s2p_spks'] # nCells, nFrames
    stimT  = data['stimT']
    twopT  = data['twopT']

    STA, STA1, STA2, r, lags, shape = compute_split_STAs_fast(
        stimpath,
        spikes,
        stimT,
        twopT,
        window=window,
        downsample=downsample,
    )

    dict_out = {
        'STA':   STA,
        'STA1':  STA1,
        'STA2':  STA2,
        'lags':  lags,
        'jcorr': r,
        'shape': np.array(shape),
        'downsample': int(downsample),
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
