# -*- coding: utf-8 -*-
"""
representational_drift/utils/analyze_movie.py

Analyze natural movie stimulus.

DMM, October 2026
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from files import find, write_h5
from twop_helpers import read_xml
from gui_funcs import select_directory
from reliability import rowwise_corr, shuffle_pvalue


def calc_split_half_reliability(resp, n_shuffles=1000, seed=0):

    nCells, nRep, nF = resp.shape
    halves = (np.arange(0, nRep, 2), np.arange(1, nRep, 2)) # even, odd repeats
    by_rep = np.ascontiguousarray(np.transpose(resp, (1, 0, 2)), dtype=np.float32) # (nRep, nCells, nF)

    r = rowwise_corr(by_rep[halves[0]].mean(axis=0), by_rep[halves[1]].mean(axis=0))

    rng = np.random.default_rng(seed)
    null = np.zeros((nCells, n_shuffles), dtype=np.float32)
    acc = np.zeros((2, nCells, nF), dtype=np.float32)
    for i in range(n_shuffles):
        shifts = rng.integers(0, nF, nRep)
        acc[:] = 0
        for h, reps in enumerate(halves):
            for rep in reps:
                k = shifts[rep]
                acc[h, :, k:] += by_rep[rep, :, :nF-k]
                acc[h, :, :k] += by_rep[rep, :, nF-k:]
        null[:, i] = rowwise_corr(acc[0], acc[1])

    return {
        'r': r,
        'p': shuffle_pvalue(r, null),
        'null': null,
    }


def main(mv_dir, time_path=None, n_shuffles=1000):

    if time_path is None:
        time_path = find(path=mv_dir, pattern='*_time*.csv', MR=True)
    mv_time = pd.read_csv(time_path)
    movie_name = os.path.basename(time_path).split('_time')[0]

    xml_data = read_xml(find(path=mv_dir, pattern='*.xml', MR=True))

    sps = np.load(os.path.join(mv_dir, 'suite2p/plane0/spks.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(mv_dir, 'suite2p/plane0/iscell.npy'), allow_pickle=True)
    sps = sps[iscell[:,0].astype(bool)]
    twopT = xml_data['rel_time']

    mv_time = mv_time.sort_values(['repeat', 'movie_frame']).reset_index(drop=True)
    nRep = mv_time['repeat'].nunique()
    nF = mv_time['movie_frame'].nunique()
    nCells = sps.shape[0]
    if len(mv_time) != nRep * nF:
        raise ValueError('Expected {} repeats x {} frames = {} rows in {}, found {}'.format(
            nRep, nF, nRep * nF, time_path, len(mv_time)))

    movie_onsets = mv_time['onset_s'].values.reshape(nRep, nF)

    movie_frame_indices = np.searchsorted(twopT, movie_onsets, side='right') - 1

    resp = sps[:, movie_frame_indices] # (nCells, nRep, nF)

    rel = calc_split_half_reliability(resp, n_shuffles=n_shuffles)
    print('{}: {}/{} cells respond reliably (p<0.05)'.format(
        movie_name, np.sum(rel['p'] < 0.05), nCells))

    movie_time = movie_onsets[0] - movie_onsets[0, 0]

    results = {
        'movie_name': movie_name,
        'resp': resp.astype(np.float32),
        'movie_time': movie_time,
        'rel_r': rel['r'],
        'rel_p': rel['p'],
        'rel_null': rel['null'],
    }
    savepath = os.path.join(mv_dir, 'movie.h5')
    write_h5(savepath, results)
    print('Wrote {}'.format(savepath))
    n_plot_cells = 8

    with PdfPages(os.path.join(mv_dir, 'movie_first_{}_cells.pdf'.format(n_plot_cells))) as pdf:
        fig, axs = plt.subplots(n_plot_cells, 2, figsize=(8.5, 11), sharex=True)
        for c in range(min(n_plot_cells, nCells)):
            axs[c,0].imshow(
                resp[c], aspect='auto', cmap='gray_r', interpolation='nearest',
                extent=[movie_time[0], movie_time[-1], nRep, 0]
            )
            axs[c,0].set_ylabel('cell {}\nrepeat'.format(c), fontsize=7)
            axs[c,0].set_title('odd/even r={:.2f}, p={:.3f}'.format(
                rel['r'][c], rel['p'][c]), fontsize=7)

            mean_resp = np.mean(resp[c], axis=0)
            sem_resp = np.std(resp[c], axis=0) / np.sqrt(nRep)
            axs[c,1].plot(movie_time, mean_resp, color='k', lw=0.75)
            axs[c,1].fill_between(movie_time, mean_resp - sem_resp, mean_resp + sem_resp,
                                  color='k', alpha=0.3, lw=0)
            axs[c,1].set_ylabel('mean spks', fontsize=7)

            for ax in axs[c]:
                ax.tick_params(labelsize=6)

        axs[-1,0].set_xlabel('movie time (s)', fontsize=7)
        axs[-1,1].set_xlabel('movie time (s)', fontsize=7)
        fig.suptitle('Movie responses', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    parser.add_argument('--time_path', default=None,
                        help='Movie timestamp csv, if it is not inside --path')
    parser.add_argument('--n_shuffles', type=int, default=1000)
    args = parser.parse_args()

    if args.path is None:
        usedir = select_directory('Select movie directory')
    else:
        usedir = args.path

    main(usedir, time_path=args.time_path, n_shuffles=args.n_shuffles)
