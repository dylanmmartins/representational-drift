# -*- coding: utf-8 -*-
"""
summary_doc.py

DMM, October 2026
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from files import find


def _read(path, keys):
    with h5py.File(path, 'r') as f:
        return {k: f[k][()] for k in keys if k in f}


def _block_mean(img, max_width=128):

    ds = int(np.ceil(img.shape[1] / max_width))
    if ds <= 1:
        return img
    H, W = img.shape[0] // ds, img.shape[1] // ds

    return img[:H*ds, :W*ds].reshape(H, ds, W, ds).mean(axis=(1, 3))


def make_summary_doc(savepath, gratings_path=None, movie_paths=(), sparse_noise_path=None,
                     cells=None, title=None, sn_shape=None):

    n_cells = {}
    gt = None
    if gratings_path is not None:
        gt = _read(gratings_path, ['orientations', 'tuning_mean', 'tuning_sem', 'tuning_even',
                                   'tuning_odd', 'osi', 'dsi', 'pref', 'rel_r', 'rel_p', 'activity_label'])
        n_cells[gratings_path] = len(gt['osi'])

    movie_files = [h5py.File(p, 'r') for p in movie_paths]
    movies = []
    for p, f in zip(movie_paths, movie_files):
        name = f['movie_name'][()]
        movies.append({
            'name': name.decode() if isinstance(name, bytes) else str(name),
            'resp': f['resp'],
            'movie_time': f['movie_time'][()],
            'rel_r': f['rel_r'][()],
            'rel_p': f['rel_p'][()],
        })
        n_cells[p] = f['resp'].shape[0]

    sn_file = None
    if sparse_noise_path is not None:
        sn_file = h5py.File(sparse_noise_path, 'r')
        if 'shape' in sn_file:
            sn_shape = tuple(sn_file['shape'][()])
        elif sn_shape is None:
            raise ValueError('{} has no stored STA shape; pass sn_shape=(H, W)'.format(sparse_noise_path))
        sn_jcorr = sn_file['jcorr'][()]
        sn_lags = sn_file['lags'][()]
        n_cells[sparse_noise_path] = sn_file['STA'].shape[0]

    if len(n_cells) == 0:
        raise ValueError('No inputs given.')
    if len(set(n_cells.values())) > 1:
        raise ValueError('Inputs have different num cells:\n' +
                         '\n'.join('  {}: {}'.format(p, n) for p, n in n_cells.items()))
    if cells is None:
        cells = range(list(n_cells.values())[0])

    rows = []
    if gt is not None:
        rows.append(('gratings', 1.4))
    rows += [('movie', 1.0)] * len(movies)
    if sn_file is not None:
        rows.append(('rf', 1.5))

    with PdfPages(savepath) as pdf:
        for c in tqdm(cells, desc='Writing pages'):
            fig = plt.figure(figsize=(8.5, 11))
            gs = fig.add_gridspec(len(rows), 1, height_ratios=[h for _, h in rows],
                                  left=0.09, right=0.95, top=0.93, bottom=0.05, hspace=0.55)
            fig.suptitle('cell {}'.format(c) if title is None else '{}  |  cell {}'.format(title, c),
                         fontsize=11)
            m = 0
            for row, (kind, _) in enumerate(rows):

                if kind == 'gratings':
                    sub = gs[row].subgridspec(1, 3, width_ratios=[2, 0.1, 1])
                    ax = fig.add_subplot(sub[0])
                    oris = gt['orientations']
                    mean, sem = gt['tuning_mean'][c], gt['tuning_sem'][c]
                    ax.plot(oris, gt['tuning_even'][c], color='tab:blue', lw=0.8, ls='--', label='even')
                    ax.plot(oris, gt['tuning_odd'][c], color='tab:orange', lw=0.8, ls='--', label='odd')
                    ax.plot(oris, mean, color='k', lw=1.5, label='all')
                    ax.fill_between(oris, mean - sem, mean + sem, color='k', alpha=0.25, lw=0)
                    ax.axvline(gt['pref'][c], color='grey', lw=0.75, ls=':')
                    ax.set_xticks(oris[::3])
                    ax.set_xlim(oris[0], oris[-1])
                    ax.set_ylim(bottom=0)
                    ax.set_xlabel('direction (deg)', fontsize=8)
                    label = gt.get('activity_label', b'mean spks')
                    ax.set_ylabel(label.decode() if isinstance(label, bytes) else str(label), fontsize=8)
                    ax.legend(fontsize=6, frameon=False, loc='upper right')
                    ax.set_title('Gratings   odd/even r = {:.2f} (p = {:.3f})'.format(
                        gt['rel_r'][c], gt['rel_p'][c]), fontsize=9, loc='left')
                    ax.tick_params(labelsize=7)

                    tax = fig.add_subplot(sub[2])
                    tax.axis('off')
                    tax.text(0, 0.5, 'OSI   {:.2f}\nDSI   {:.2f}\npref  {:.0f} deg\nr     {:.2f}\np     {:.3f}'.format(
                        gt['osi'][c], gt['dsi'][c], gt['pref'][c], gt['rel_r'][c], gt['rel_p'][c]),
                        fontsize=10, family='monospace', va='center')

                elif kind == 'movie':
                    mv = movies[m]
                    m += 1
                    resp = mv['resp'][c]
                    t = mv['movie_time']
                    ax = fig.add_subplot(gs[row])
                    vmax = np.percentile(resp, 99.5)
                    ax.imshow(resp, aspect='auto', cmap='Greys', interpolation='nearest',
                              vmin=0, vmax=vmax if vmax > 0 else None,
                              extent=[t[0], t[-1], resp.shape[0], 0])
                    ax.set_title('{}   odd/even r = {:.2f} (p = {:.3f})'.format(
                        mv['name'], mv['rel_r'][c], mv['rel_p'][c]), fontsize=9, loc='left')
                    ax.set_ylabel('repeat', fontsize=8)
                    ax.set_xlabel('movie time (s)', fontsize=8)
                    ax.tick_params(labelsize=7)

                elif kind == 'rf':
                    stas = [_block_mean(sn_file[k][c].reshape(sn_shape)) for k in ('STA1', 'STA2', 'STA')]
                    clim = max(np.max(np.abs(s)) for s in stas)
                    sub = gs[row].subgridspec(1, 3, wspace=0.08)
                    for i, (sta, name) in enumerate(zip(stas, ('first half', 'second half', 'full'))):
                        ax = fig.add_subplot(sub[i])
                        ax.imshow(sta, cmap='RdBu_r', vmin=-clim, vmax=clim, interpolation='nearest')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xlabel(name, fontsize=8)
                        if i == 0:
                            ax.set_title('Sparse noise RF   Jaccard = {:.2f}, lag = {} frames'.format(
                                sn_jcorr[c], sn_lags[c]), fontsize=9, loc='left')

            pdf.savefig(fig)
            plt.close(fig)

    for f in movie_files:
        f.close()
    if sn_file is not None:
        sn_file.close()
    print('Wrote {}'.format(savepath))


def find_session_outputs(session_dir):
    gratings_path = find('gratings.h5', session_dir, MR=True, retempty=True)
    movie_paths = sorted(find('movie.h5', session_dir, retempty=True) or [])
    sparse_noise_path = find('sparse_noise.h5', session_dir, MR=True, retempty=True)
    return gratings_path, movie_paths, sparse_noise_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default=None,
                        help='Session directory to search for gratings.h5, movie.h5 and sparse_noise.h5')
    parser.add_argument('--gratings', default=None)
    parser.add_argument('--movies', nargs='*', default=None)
    parser.add_argument('--sparse_noise', default=None)
    parser.add_argument('--out', default=None, help='Output PDF (default: <session>/cell_summary.pdf)')
    parser.add_argument('--cells', type=int, nargs='*', default=None,
                        help='Cell indices to include (default: all)')
    args = parser.parse_args()

    gratings_path, movie_paths, sparse_noise_path = None, [], None
    if args.session is not None:
        gratings_path, movie_paths, sparse_noise_path = find_session_outputs(args.session)
    if args.gratings is not None:
        gratings_path = args.gratings
    if args.movies is not None:
        movie_paths = args.movies
    if args.sparse_noise is not None:
        sparse_noise_path = args.sparse_noise

    out = args.out
    if out is None:
        out_dir = args.session if args.session is not None else os.getcwd()
        out = os.path.join(out_dir, 'cell_summary.pdf')
    title = os.path.basename(os.path.normpath(args.session)) if args.session is not None else None

    print('gratings: {}\nmovies: {}\nsparse noise: {}'.format(gratings_path, movie_paths, sparse_noise_path))
    make_summary_doc(out, gratings_path=gratings_path, movie_paths=movie_paths,
                     sparse_noise_path=sparse_noise_path, cells=args.cells, title=title)
