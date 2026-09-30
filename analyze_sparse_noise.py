# -*- coding: utf-8 -*-
"""
analyze_sparse_noise.py

Preprocess a sparse-noise two-photon recording (suite2p outputs + Bruker
timestamps + psychopy stimulus timestamps), write <rec>_preproc.h5, and
compute split-half receptive-field STAs from it.

Usage
-----
    python analyze_sparse_noise.py /path/to/recording_dir
    python analyze_sparse_noise.py /path/to/recording_dir --plot-only

--plot-only skips the analysis and writes the STA PDF from an existing
sparse_noise.h5.

Run from the repo root in an env with oasis + h5py (e.g. fm2/fm3). OMSI spike
inference is run out-of-process in the `spikeinf` conda env.
"""

import os
import argparse
import numpy as np
# NumPy version shim: files saved with NumPy 2.x reference numpy._core,
# but NumPy 1.x only has numpy.core.  Register aliases so pickle can find them.
import sys as _sys_np_shim
_sys_np_shim.modules.setdefault('numpy._core', np.core)
for _a in dir(np.core):
    _sys_np_shim.modules.setdefault(f'numpy._core.{_a}', getattr(np.core, _a))
del _sys_np_shim, _a
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from twop import TwoP, run_omsi
from twop_helpers import read_xml
from utils.files import find, write_h5
from sparse_noise_mapping import calc_sparse_noise_STA_reliability, default_stimpath


cfg = {
    # Fallback 2P frame rate (Hz); replaced by the rate in the Bruker XML when
    # it is available. TwoP only uses it to set self.dt.
    'twop_rate': 7.49,
    # SD threshold used by TwoP.calc_dFF_transients and TwoP.normalize_spikes.
    'cell_sd_thresh': 2,
    # Neuropil subtraction coefficient for dF/F and OMSI.
    'neu_correction': 0.7,
    # Discrete spike-time inference: 'omsi' (MCMC, falls back to 'oasis' if
    # the conda env is unavailable) or 'oasis' (thresholded OASIS AR1).
    'spike_method': 'omsi',
    'omsi_env': 'spikeinf',
    'oasis_spike_thresh': 0.05,
    # Filename patterns of the psychopy timestamp file, in order of preference.
    'stimT_patterns': ['*sparsenoise.csv', 'sparse_noise_time*.txt'],
    # Columns of the psychopy timestamp file to use as stimulus onset times, in
    # order of preference. 'psychopy_time' is what the recorded sessions have;
    # 'stim_onset_clock' is what sparse_noise_v01.py writes; 'onset_s' is what
    # the sparse_noise_time_*.txt files have.
    'stimT_columns': ['psychopy_time', 'stim_onset_clock', 'onset_s'],
    # Offset (s) added to the stimulus times to put them on the 2P clock. The
    # psychopy and 2P clocks are not synchronized; None estimates it with a
    # low-resolution split-half STA lag scan (see estimate_stim_lag).
    'stim_lag': None,
    'lag_scan_range': (-10., 30.),
    'lag_scan_step': 0.25,
    # Number of cells in the STA PDF.
    'n_plot_cells': 8,
    # Stimulus frame period (s) used to make up timestamps when no
    # *sparsenoise.csv exists. Nominal on_time is 0.5 s, but frames run long:
    # measured mean interval in 251016_DMM_DMM061_pos18/sn1 was 0.5295 s
    # (4000 frames over 2117.6 s). Using 0.5 would drift ~118 s by the end.
    'stim_frame_period': 0.5295,
    # Sparse noise stimulus array (N, H, W[, C]); None uses the OS default in
    # sparse_noise_mapping.
    'stimpath': None,
    # STA lag window in stimulus frames.
    'sta_window': 13,
}


def find_bruker_xml(rpath):
    """ Return the main Bruker T-series XML, skipping voltage/reference files. """

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
    """ Stimulus onset times (s) from the psychopy timestamp file, or made up if absent.

    If no file matching cfg['stimT_patterns'] exists in `rpath`, assumes one onset per stimulus
    frame at a constant `cfg['stim_frame_period']`, starting at t=0. This
    ignores frame-to-frame jitter and any slow drift in the true period, so
    it is only approximately aligned (in the measured sn1 session a constant
    period was up to ~1 s off mid-recording).
    """

    n_stim = np.load(stimpath, mmap_mode='r').shape[0]

    csv_path = None
    for pattern in cfg['stimT_patterns']:
        csv_path = find(pattern, rpath, MR=True, retempty=True)
        if csv_path is not None:
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


def estimate_stim_lag(stimpath, spks, twopT, stimT, cfg, block=16):
    """ Offset (s) to add to stimT to put it on the 2P clock.

    Scans lags and picks the one where STAs from the first and second half of
    the recording agree best (median correlation across cells). The sparse
    noise sequence does not repeat, so a wrong lag gives uncorrelated halves.
    Uses a block-averaged stimulus and each cell's mean activity during each
    stimulus frame, so it takes ~30 s instead of the full STA computation.
    """

    stim_mmap = np.load(stimpath, mmap_mode='r')
    n_stim, H, W = stim_mmap.shape[:3]
    Hc, Wc = (H // block) * block, (W // block) * block
    stim_ds = np.zeros([n_stim, Hc // block, Wc // block], dtype=np.float32)
    print('  Block-averaging stimulus for lag scan...')
    for s in range(0, n_stim, 50):
        chunk = stim_mmap[s:s+50, :Hc, :Wc]
        if chunk.ndim == 4:
            chunk = chunk[..., 0]
        chunk = np.asarray(chunk, dtype=np.float32)
        stim_ds[s:s+50] = chunk.reshape(-1, Hc // block, block, Wc // block, block).mean(axis=(2, 4))
    stim_ds = stim_ds.reshape(n_stim, -1)
    stim_ds -= stim_ds.mean(axis=0)

    frame_dur = np.nanmedian(np.diff(stimT))
    spks_cumsum = np.concatenate([np.zeros([spks.shape[0], 1]), np.cumsum(spks, axis=1)], axis=1)

    def split_half_reliability(lag):
        # mean activity of each cell during each stimulus frame, shifted by lag
        onsets = stimT + lag
        i0 = np.searchsorted(twopT, onsets)
        i1 = np.searchsorted(twopT, onsets + frame_dur)
        valid = (onsets >= twopT[0]) & (onsets + frame_dur <= twopT[-1]) & (i1 > i0)
        half1 = valid & (np.arange(n_stim) < n_stim // 2)
        half2 = valid & (np.arange(n_stim) >= n_stim // 2)
        if np.sum(half1) < n_stim // 4 or np.sum(half2) < n_stim // 4:
            return np.nan # too little of this half falls inside the recording
        rates = np.zeros([spks.shape[0], n_stim], dtype=np.float32)
        rates[:, valid] = (spks_cumsum[:, i1[valid]] - spks_cumsum[:, i0[valid]]) / (i1 - i0)[valid]
        stas = []
        for half in (half1, half2):
            r = rates[:, half] - np.mean(rates[:, half], axis=1, keepdims=True)
            sta = r @ stim_ds[half]
            stas.append(sta - np.mean(sta, axis=1, keepdims=True))
        corr = np.sum(stas[0] * stas[1], axis=1) / np.sqrt(
            np.sum(stas[0]**2, axis=1) * np.sum(stas[1]**2, axis=1))
        return np.nanmedian(corr)

    lags = np.arange(cfg['lag_scan_range'][0], cfg['lag_scan_range'][1] + cfg['lag_scan_step'] / 2,
                     cfg['lag_scan_step'])
    lag_reliability = np.array([split_half_reliability(lag) for lag in lags])
    best_lag = lags[np.nanargmax(lag_reliability)]
    rel_at_zero = split_half_reliability(0.)

    if np.abs(best_lag) >= np.nanmedian(np.diff(twopT)):
        print('*' * 70)
        print('Stimulus/2P timing offset detected: {:.2f} s (scanned {:.2f} to {:.2f} s)'.format(
            best_lag, lags[0], lags[-1]))
        print('Median split-half STA correlation: {:.3f} at 0 s, {:.3f} at {:.2f} s'.format(
            rel_at_zero, np.nanmax(lag_reliability), best_lag))
        print('Stimulus times are shifted by this lag for the analysis below.')
        print('Other stimuli recorded on this day may have the same sync problem.')
        if best_lag in (lags[0], lags[-1]):
            print('Best lag is at the edge of the scan range, so the true offset may be larger.')
        print('*' * 70)

    return best_lag, lags, lag_reliability


def plot_split_STAs(sn_path, stimpath, pdf_path, n_cells=8):
    """ PDF of first-half, second-half and full STAs for the first n_cells cells.

    Reads only those cells' rows from sparse_noise.h5, which is too large to load whole.
    """

    H, W = np.load(stimpath, mmap_mode='r').shape[1:3]
    with h5py.File(sn_path, 'r') as f:
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


def get_discrete_spike_times_oasis(raw_F, raw_Fneu, fs, neu_correction=0.7,
                                   spike_thresh=0.05):
    """ OASIS AR1 deconvolution, thresholded to discrete spike times (s).

    Uses an 8th-percentile baseline dF/F so the result is on the same scale as
    OMSI's internal dF/F.
    """

    from oasis.functions import deconvolve

    Fc = np.asarray(raw_F, dtype=float) - neu_correction * np.asarray(raw_Fneu, dtype=float)
    F0 = np.percentile(Fc, 8, axis=1, keepdims=True)
    dff = (Fc - F0) / np.abs(F0)

    g = np.exp(-1 / (fs * 0.5))
    spike_times = []
    for i in range(np.size(dff, 0)):
        _, s, _, _, _ = deconvolve(dff[i], g=(g,), penalty=1)
        spike_times.append(np.where(s > spike_thresh)[0] / fs)

    return spike_times


def get_spike_times_list(raw_F, raw_Fneu, fs, cfg):
    """ Per-cell discrete spike times in seconds from frame 0 of the recording.

    Returns
    -------
    spike_times : list of np.ndarray, length n_cells
    prob_trace : (n_cells, n_frames) array or None
        OMSI expected spikes per frame; None when OASIS was used.
    """

    if cfg['spike_method'] == 'omsi':
        print('  OMSI spike inference on {} cells at {:.3f} Hz...'.format(
            np.size(raw_F, 0), fs))
        omsi = run_omsi(raw_F, raw_Fneu, fs, env=cfg['omsi_env'],
                        f_corr=cfg['neu_correction'])
        if omsi['spike_times'] is not None:
            return omsi['spike_times'], omsi['prob_trace']
        print('  OMSI unavailable, falling back to OASIS spike times.')

    print('  OASIS spike inference on {} cells...'.format(np.size(raw_F, 0)))
    spike_times = get_discrete_spike_times_oasis(
        raw_F, raw_Fneu, fs,
        neu_correction=cfg['neu_correction'],
        spike_thresh=cfg['oasis_spike_thresh']
    )
    return spike_times, None


def main(rpath, cfg=cfg):

    rpath = os.path.abspath(rpath)
    full_rname = os.path.split(rpath)[1]

    stimpath = cfg['stimpath'] if cfg['stimpath'] is not None else default_stimpath()
    xml_path = find_bruker_xml(rpath)

    F = np.load(find('F.npy', rpath, MR=True), allow_pickle=True)
    Fneu = np.load(find('Fneu.npy', rpath, MR=True), allow_pickle=True)
    spks = np.load(find('spks.npy', rpath, MR=True), allow_pickle=True)
    iscell = np.load(find('iscell.npy', rpath, MR=True), allow_pickle=True)
    stat = np.load(find('stat.npy', rpath, MR=True), allow_pickle=True)
    ops = np.load(find('ops.npy', rpath, MR=True), allow_pickle=True)

    # Read Bruker timestamps. relativeTime is seconds since the first frame,
    # i.e. since the trigger that started acquisition.
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
    twop_dict = twop_recording.calc_dFF(neu_correction=cfg['neu_correction'], use_oasis=True)
    dFF_transients = twop_recording.calc_dFF_transients()
    # Set a maximum spike rate for each cell, then normalize spikes
    normspikes = twop_recording.normalize_spikes()
    recording_props = twop_recording.get_recording_props(
        stat=stat,
        ops=ops
    )

    # Bruker can log one more (or fewer) frame than suite2p processed.
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
    stim_lag = cfg['stim_lag']
    if stim_lag is None:
        stim_lag, _, _ = estimate_stim_lag(
            stimpath, spks[iscell[:,0].astype(bool)][:, :len(twopT)], twopT, sn_stimT, cfg)
    sn_stimT = sn_stimT + stim_lag

    preprocessed_dict = {**twop_dict, **recording_props}
    preprocessed_dict['stimT'] = sn_stimT
    preprocessed_dict['stim_lag'] = float(stim_lag)

    # Discrete spike times, stored as a NaN-padded (n_cells, max_spikes) array
    # so it can be written to HDF5.
    _fs = float(1.0 / np.nanmedian(np.diff(twopT)))
    _spike_times_list, _prob_trace = get_spike_times_list(
        twop_dict['raw_F'], twop_dict['raw_Fneu'], _fs, cfg
    )
    if _prob_trace is not None:
        preprocessed_dict['omsi_prob_trace'] = np.asarray(_prob_trace, dtype=np.float32)

    _n_cells = len(_spike_times_list)
    _max_sp = max((len(st) for st in _spike_times_list), default=1)
    _sp_array = np.full((_n_cells, max(_max_sp, 1)), np.nan, dtype=np.float64)
    for _i, _st in enumerate(_spike_times_list):
        _sp_array[_i, :len(_st)] = _st
    preprocessed_dict['spike_times'] = _sp_array

    _savepath = os.path.join(rpath, '{}_preproc.h5'.format(full_rname))
    print('  Writing preprocessed data to {}...'.format(_savepath))
    write_h5(_savepath, preprocessed_dict)

    calc_sparse_noise_STA_reliability(
        _savepath,
        stimpath=stimpath,
        window=cfg['sta_window']
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
        main(args.rpath)
