# -*- coding: utf-8 -*-
"""
representational_drift/utils/analyze_sparse_noise.py

Analyze sparse noise stimulus.

DMM, October 2026
"""

import os
import glob
import argparse
import numpy as np
import sys as _sys_np_shim
_sys_np_shim.modules.setdefault('numpy._core', np.core)
for _a in dir(np.core):
    _sys_np_shim.modules.setdefault(f'numpy._core.{_a}', getattr(np.core, _a))
del _sys_np_shim, _a
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from twop import TwoP
from twop_helpers import read_xml
from files import find, write_h5
from sparse_noise_mapping import calc_sparse_noise_STA_reliability, default_stimpath
from sparse_noise import MAX_STA_LAG
from activity import load_activity, SOURCES, DEFAULT_SOURCE


cfg = {
    'twop_rate': 7.49,
    'cell_sd_thresh': 2,
    'neu_correction': 0.7,
    'activity_source': DEFAULT_SOURCE,
    'omsi_env': 'spikeinf',
    'stimT_patterns': ['*sparsenoise.csv', 'sparse_noise_timestamps_*.csv', 'sparse_noise_time*.txt'],
    'stimT_columns': ['psychopy_time', 'stim_onset_clock', 'onset_s'],
    'n_plot_cells': 8,
    'stim_frame_period': 0.5295,
    'stimpath': None,
    'sta_window': MAX_STA_LAG,
    'sta_downsample': 4,
}


def find_bruker_xml(rpath):

    xmls = find('*.xml', rpath)
    xmls = [
        x for x in xmls
        if ('Voltage' not in os.path.basename(x))
        and ('References' not in x)
    ]
    if len(xmls) == 0:
        raise FileNotFoundError('No Bruker T-series XML found in {}'.format(rpath))
    return max(xmls, key=os.path.getmtime)


def load_stimT(rpath, stimpath, cfg):

    n_stim = np.load(stimpath, mmap_mode='r').shape[0]

    csv_path = None
    for pattern in cfg['stimT_patterns']:
        csv_path = find(pattern, rpath, MR=True, retempty=True)
        if csv_path is not None:
            break
    if csv_path is None:
        for pattern in cfg['stimT_patterns']:
            matches = glob.glob(os.path.join(os.path.dirname(rpath), pattern))
            if len(matches) > 0:
                csv_path = max(matches, key=os.path.getmtime)
                break

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        for col in cfg['stimT_columns']:
            if col in df.columns:
                print('  Stimulus timestamps from {} ({!r})'.format(csv_path, col))
                stimT = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
                if np.sum(np.isfinite(stimT)) != n_stim:
                    raise ValueError('{} has {} valid {!r} timestamps but the stimulus has {} frames'.format(
                        csv_path, np.sum(np.isfinite(stimT)), col, n_stim))
                return stimT
        raise KeyError('{} has none of the columns {}; found {}'.format(
            csv_path, cfg['stimT_columns'], list(df.columns)))

    period = cfg['stim_frame_period']
    print('  WARNING: no stimulus timestamp file ({}) in {}; making up {} stimulus '
          'timestamps at a constant {:.4f} s period. Alignment is '
          'approximate.'.format(', '.join(cfg['stimT_patterns']), rpath, n_stim, period))
    return np.arange(n_stim) * period


def plot_split_STAs(sn_path, stimpath, pdf_path, n_cells=8):

    with h5py.File(sn_path, 'r') as f:
        if 'shape' in f:
            H, W = f['shape'][()]
        else:
            H, W = np.load(stimpath, mmap_mode='r').shape[1:3]
        n_cells = min(n_cells, f['STA'].shape[0])
        stas = [f[k][:n_cells].reshape(n_cells, H, W) for k in ('STA1', 'STA2', 'STA')]
        jcorr = f['jcorr'][:n_cells]

    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(n_cells, 3, figsize=(8.5, 11), squeeze=False)
        for c in range(n_cells):
            clim = np.max([np.max(np.abs(sta[c])) for sta in stas])
            for ax, sta, title in zip(axs[c], stas, ('first half', 'second half', 'full')):
                ax.imshow(sta[c], cmap='RdBu_r', vmin=-clim, vmax=clim, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_title(title, fontsize=8)
            axs[c,0].set_ylabel('cell {}\nJaccard={:.2f}'.format(c, jcorr[c]), fontsize=7)
        fig.suptitle('Sparse noise STAs: {}'.format(os.path.dirname(sn_path)), fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        pdf.savefig(fig)
        plt.close(fig)
    print('  Wrote {}'.format(pdf_path))


def main(rpath, cfg=cfg):

    rpath = os.path.abspath(rpath)
    full_rname = os.path.split(rpath)[1]

    stimpath = cfg['stimpath'] if cfg['stimpath'] is not None else default_stimpath()
    xml_path = find_bruker_xml(rpath)

    F_path = find('F.npy', rpath, MR=True)
    F = np.load(F_path, allow_pickle=True)
    Fneu = np.load(find('Fneu.npy', rpath, MR=True), allow_pickle=True)
    spks = np.load(find('spks.npy', rpath, MR=True), allow_pickle=True)
    iscell = np.load(find('iscell.npy', rpath, MR=True), allow_pickle=True)
    stat = np.load(find('stat.npy', rpath, MR=True), allow_pickle=True)
    ops = np.load(find('ops.npy', rpath, MR=True), allow_pickle=True)

    print('  Reading Bruker timestamps from {}'.format(xml_path))
    acq_props = read_xml(xml_path)
    twopT = acq_props['rel_time']
    cfg = dict(cfg)
    if acq_props['acq_Hz'] is not None:
        cfg['twop_rate'] = acq_props['acq_Hz']

    twop_recording = TwoP(rpath, full_rname, cfg=cfg)
    twop_recording.add_data(
        F=F,
        Fneu=Fneu,
        spikes=spks,
        iscell=iscell
    )
    twop_dict = twop_recording.calc_dFF(neu_correction=cfg['neu_correction'], use_oasis=False)
    dFF_transients = twop_recording.calc_dFF_transients()
    normspikes = twop_recording.normalize_spikes()
    recording_props = twop_recording.get_recording_props(
        stat=stat,
        ops=ops
    )

    n_frames = np.size(twop_dict['s2p_spks'], 1)
    if len(twopT) != n_frames:
        print('  {} Bruker timestamps vs {} suite2p frames; trimming to match.'.format(
            len(twopT), n_frames))
        if len(twopT) < n_frames:
            raise ValueError('Fewer Bruker timestamps than suite2p frames.')
        twopT = twopT[:n_frames]

    twop_dict['twopT'] = twopT
    twop_dict['twop_rate'] = float(cfg['twop_rate'])
    twop_dict['matlab_cellinds'] = np.arange(np.size(twop_dict['raw_F'], 0))
    twop_dict['norm_spikes'] = normspikes
    twop_dict['dFF_transients'] = dFF_transients

    sn_stimT = load_stimT(rpath, stimpath, cfg)

    preprocessed_dict = {**twop_dict, **recording_props}
    preprocessed_dict['stimT'] = sn_stimT

    _fs = float(1.0 / np.nanmedian(np.diff(twopT)))
    preprocessed_dict['activity'] = load_activity(
        os.path.dirname(F_path), _fs,
        source=cfg['activity_source'],
        neu_correction=cfg['neu_correction'],
        omsi_env=cfg['omsi_env']
    )
    preprocessed_dict['activity_source'] = cfg['activity_source']

    _savepath = os.path.join(rpath, '{}_preproc.h5'.format(full_rname))
    print('  Writing preprocessed data to {}...'.format(_savepath))
    write_h5(_savepath, preprocessed_dict)

    calc_sparse_noise_STA_reliability(
        _savepath,
        stimpath=stimpath,
        window=cfg['sta_window'],
        downsample=cfg['sta_downsample']
    )

    plot_split_STAs(
        os.path.join(rpath, 'sparse_noise.h5'),
        stimpath,
        os.path.join(rpath, 'sparse_noise_STAs_first_{}_cells.pdf'.format(cfg['n_plot_cells'])),
        n_cells=cfg['n_plot_cells']
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('rpath', type=str, help='Recording directory')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only write the STA PDF from an existing sparse_noise.h5')
    parser.add_argument('--activity', choices=SOURCES, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    if args.plot_only:
        rpath = os.path.abspath(args.rpath)
        plot_split_STAs(
            find('sparse_noise.h5', rpath, MR=True),
            cfg['stimpath'] if cfg['stimpath'] is not None else default_stimpath(),
            os.path.join(rpath, 'sparse_noise_STAs_first_{}_cells.pdf'.format(cfg['n_plot_cells'])),
            n_cells=cfg['n_plot_cells']
        )
    else:
        main(args.rpath, cfg=dict(cfg, activity_source=args.activity))
