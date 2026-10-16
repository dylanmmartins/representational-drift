# -*- coding: utf-8 -*-
"""
utils/sparse_noise_fast.py

Split-half sparse-noise STAs for all cells at once as matrix products.

DMM, October 2026
"""

import time
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from sparse_noise import jaccard_topk

_STIM_CHANNEL = 0


def load_signed_stimulus(stimpath, downsample=1, batch_frames=64):

    stim_mmap = np.load(stimpath, mmap_mode='r')
    has_channel = stim_mmap.ndim == 4
    n_stim, H, W = stim_mmap.shape[:3]
    ds = int(downsample)
    Hd, Wd = H // ds, W // ds
    n_features = Hd * Wd

    def read(idx):
        chunk = stim_mmap[idx, :, :, _STIM_CHANNEL] if has_channel else stim_mmap[idx]
        return np.asarray(chunk)[:, :Hd*ds, :Wd*ds]

    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(n_stim, min(512, n_stim), replace=False))
    bg_est = float(np.median(read(sample_idx)))

    print('  Stimulus: {} frames, {}x{} px, downsample {} -> {}x{} = {} pixels (bg={:.1f})'.format(
        n_stim, H, W, ds, Hd, Wd, n_features, bg_est))

    dtype = np.int8 if ds == 1 else np.float32
    flat = np.empty((n_stim, n_features), dtype=dtype)

    for s in tqdm(range(0, n_stim, batch_frames), desc='  Reading stimulus'):
        e = min(s + batch_frames, n_stim)
        chunk = read(slice(s, e))
        signed = (chunk > bg_est).astype(np.int8) - (chunk < bg_est).astype(np.int8)
        if ds > 1:
            signed = signed.reshape(e - s, Hd, ds, Wd, ds).mean(axis=(2, 4), dtype=np.float32)
        flat[s:e] = signed.reshape(e - s, n_features)

    col_means = flat.mean(axis=0, dtype=np.float64).astype(np.float32)

    return flat, col_means, (Hd, Wd)


def compute_split_STAs_fast(stimpath, spikes, stim_times, spike_times,
                            window=13, downsample=1, max_block_bytes=2**29):

    t0 = time.time()
    spikes = np.asarray(spikes, dtype=np.float32)
    stim_times = np.asarray(stim_times, dtype=np.float64)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    n_cells = spikes.shape[0]
    eps = 1e-9

    flat, col_means, shape = load_signed_stimulus(stimpath, downsample=downsample)
    n_stim, n_features = flat.shape
    if len(stim_times) != n_stim:
        raise ValueError('{} stimulus timestamps but {} stimulus frames'.format(
            len(stim_times), n_stim))

    spikeend = int(np.searchsorted(spike_times, stim_times[-1], side='right'))
    spike_times_trim = spike_times[:spikeend]
    rates = interp1d(spike_times_trim, spikes[:, :spikeend], axis=1, kind='linear',
                     fill_value='extrapolate', assume_sorted=True)(stim_times)
    rates = np.maximum(rates, 0).astype(np.float32)

    midpoint_time = float(spike_times_trim[spike_times_trim.size // 2])
    stim_split_ind = int(np.clip(np.searchsorted(stim_times, midpoint_time, side='right'),
                                 1, n_stim - 1))
    print('  Split at stim frame {}/{} (t={:.1f}s)'.format(stim_split_ind, n_stim, midpoint_time))

    T = n_stim - 1 - window
    resp_inds = np.arange(window, n_stim - 1)
    R = rates[:, resp_inds]
    R12 = np.zeros((2 * n_cells, T), dtype=np.float32)
    first = resp_inds < stim_split_ind
    R12[:n_cells, first] = R[:, first]
    R12[n_cells:, ~first] = R[:, ~first]
    tot12 = R12.sum(axis=1, dtype=np.float64)
    tot = tot12[:n_cells] + tot12[n_cells:]
    norm1 = (1 / (tot12[:n_cells] + eps)).astype(np.float32)[:, None]
    norm2 = (1 / (tot12[n_cells:] + eps)).astype(np.float32)[:, None]
    norm = (1 / (tot + eps)).astype(np.float32)[:, None]

    block = int(np.clip(max_block_bytes // (4 * 2 * n_cells), 256, n_features))
    blocks = [(s, min(s + block, n_features)) for s in range(0, n_features, block)]
    n_lags = window + 1

    def centered(s, e):
        return flat[:, s:e].astype(np.float32) - col_means[s:e]

    print('  {} cells, {} lags, {} pixel blocks of {}'.format(n_cells, n_lags, len(blocks), block))

    peak = np.zeros((3, n_cells, n_lags), dtype=np.float32) # full, first, second
    for s, e in tqdm(blocks, desc='  Pass 1/2 (best lag)'):
        Sc = centered(s, e)
        for k in range(n_lags):
            U = R12 @ Sc[k:k+T]
            U1, U2 = U[:n_cells], U[n_cells:]
            np.maximum(peak[0, :, k], np.abs((U1 + U2) * norm).max(axis=1), out=peak[0, :, k])
            np.maximum(peak[1, :, k], np.abs(U1 * norm1).max(axis=1), out=peak[1, :, k])
            np.maximum(peak[2, :, k], np.abs(U2 * norm2).max(axis=1), out=peak[2, :, k])
    best_k = np.argmax(peak, axis=2) # (3, n_cells)

    R_full = R12[:n_cells] + R12[n_cells:]
    weights = ((R_full, norm), (R12[:n_cells], norm1), (R12[n_cells:], norm2))
    out = np.zeros((3, n_cells, n_features), dtype=np.float32)
    for s, e in tqdm(blocks, desc='  Pass 2/2 (STAs)'):
        Sc = centered(s, e)
        for k in range(n_lags):
            for j, (W_, nrm) in enumerate(weights):
                rows = np.where(best_k[j] == k)[0]
                if len(rows) > 0:
                    out[j, rows, s:e] = (W_[rows] @ Sc[k:k+T]) * nrm[rows]

    STA, STA1, STA2 = out
    split_corr = np.array([jaccard_topk(STA1[c], STA2[c]) for c in range(n_cells)])
    best_lags = window - best_k[0]

    print('  STAs done in {:.1f} s'.format(time.time() - t0))

    return STA, STA1, STA2, split_corr, best_lags, shape
