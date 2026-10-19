# -*- coding: utf-8 -*-
"""
representational_drift/utils/analyze_gratings.py

Analyze gratings stimulus.

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
from activity import load_activity, frame_rate, SOURCES, DEFAULT_SOURCE, LABELS


def calc_split_half_reliability(trial_resp, trial_oris, orientations, n_shuffles=1000, seed=0):

    n_trials = len(trial_oris)
    n_oris = len(orientations)

    avg = np.zeros((2, n_trials, n_oris))
    for o, ori in enumerate(orientations):
        inds = np.where(trial_oris == ori)[0]
        for half in (0, 1):
            half_inds = inds[half::2]
            if len(half_inds) > 0:
                avg[half, half_inds, o] = 1 / len(half_inds)

    use_oris = np.all(avg.sum(axis=1) > 0, axis=0)
    if not np.all(use_oris):
        print('Warning: orientations {} lack an odd or even presentation; left out of reliability'.format(
            orientations[~use_oris]))
    avg = avg[:, :, use_oris]

    def split_corr(resp):
        return rowwise_corr(resp @ avg[0], resp @ avg[1])

    r = split_corr(trial_resp)

    rng = np.random.default_rng(seed)
    null = np.zeros((trial_resp.shape[0], n_shuffles))
    for i in range(n_shuffles):
        null[:, i] = split_corr(trial_resp[:, rng.permutation(n_trials)])

    tuning_even = np.full((trial_resp.shape[0], n_oris), np.nan)
    tuning_odd = np.full((trial_resp.shape[0], n_oris), np.nan)
    tuning_even[:, use_oris] = trial_resp @ avg[0]
    tuning_odd[:, use_oris] = trial_resp @ avg[1]

    return {
        'r': r,
        'p': shuffle_pvalue(r, null),
        'null': null,
        'tuning_even': tuning_even,
        'tuning_odd': tuning_odd,
    }


def main(gt_dir, time_path=None, n_shuffles=1000, activity_source=DEFAULT_SOURCE):

    if time_path is None:
        time_path = find(path=gt_dir, pattern='gratings_timestamps*.csv', MR=True)
    gt_time = pd.read_csv(time_path)
    gt_time = gt_time.sort_values('onset_s').reset_index(drop=True) # presentation order

    xml_data = read_xml(find(path=gt_dir, pattern='*.xml', MR=True))

    twopT = xml_data['rel_time']
    sps = load_activity(os.path.join(gt_dir, 'suite2p/plane0'), frame_rate(twopT), source=activity_source)

    orientations = np.arange(0, 360, 30)
    trial_oris = gt_time['orientation_deg'].astype(int).values

    missing = set(orientations) - set(trial_oris)
    if missing:
        print('Warning: no presentations of orientations {}'.format(sorted(missing)))
    unexpected = set(trial_oris) - set(orientations)
    if unexpected:
        print('Warning: presented orientations {} not in analysis orientations'.format(sorted(unexpected)))

    trial_resp = np.zeros([sps.shape[0], len(gt_time)]) * np.nan
    for i in range(len(gt_time)):
        start = gt_time['onset_s'][i]
        end = gt_time['offset_s'][i]
        spike_indices = np.where((twopT >= start) & (twopT <= end))[0]
        if len(spike_indices) > 0:
            trial_resp[:,i] = np.nanmean(sps[:, spike_indices], axis=1)

    gratings_data = np.zeros([sps.shape[0], len(orientations), 2]) * np.nan
    for o, ori in enumerate(orientations):
        frs = trial_resp[:, trial_oris==ori]
        if frs.shape[1] > 0:
            gratings_data[:,o,0] = np.nanmean(frs, axis=1)
            gratings_data[:,o,1] = np.nanstd(frs, axis=1) / np.sqrt(frs.shape[1])

    def calc_gratings_selectivity(tuning):
        n_dirs = len(tuning)
        half_turn = n_dirs // 2
        quarter_turn = n_dirs // 4

        th_pref = np.nanargmax(tuning, axis=0)
        th_null = (th_pref + half_turn) % n_dirs
        th_ortho = (th_pref + quarter_turn) % n_dirs

        R_pref_ori = (tuning[th_pref] + tuning[th_null]) * 0.5
        R_ortho = (tuning[th_ortho] + tuning[(th_ortho + half_turn) % n_dirs]) * 0.5
        R_pref_dir = tuning[th_pref]
        R_null = tuning[th_null]

        osi = (R_pref_ori - R_ortho) / (R_pref_ori + R_ortho)
        dsi = (R_pref_dir - R_null) / (R_pref_dir + R_null)

        return osi, dsi, orientations[th_pref]


    osis = np.zeros(sps.shape[0]) * np.nan
    dsis = np.zeros(sps.shape[0]) * np.nan
    prefs = np.zeros(sps.shape[0]) * np.nan

    for c in range(sps.shape[0]):
        osis[c], dsis[c], prefs[c] = calc_gratings_selectivity(gratings_data[c, :, 0])

    has_frames = ~np.isnan(trial_resp[0])
    rel = calc_split_half_reliability(
        trial_resp[:, has_frames], trial_oris[has_frames], orientations, n_shuffles=n_shuffles)
    print('{}/{} cells have reliable tuning (p<0.05)'.format(
        np.sum(rel['p'] < 0.05), sps.shape[0]))

    results = {
        'orientations': orientations,
        'tuning_mean': gratings_data[:,:,0],
        'tuning_sem': gratings_data[:,:,1],
        'osi': osis,
        'dsi': dsis,
        'pref': prefs,
        'rel_r': rel['r'],
        'rel_p': rel['p'],
        'rel_null': rel['null'].astype(np.float32),
        'tuning_even': rel['tuning_even'],
        'tuning_odd': rel['tuning_odd'],
        'trial_resp': trial_resp,
        'trial_oris': trial_oris,
        'activity_source': activity_source,
        'activity_label': LABELS[activity_source],
    }
    savepath = os.path.join(gt_dir, 'gratings.h5')
    write_h5(savepath, results)
    print('Wrote {}'.format(savepath))


    fig, axs = plt.subplots(1, 3, figsize=(9,2), dpi=300)
    axs[0].hist(osis, bins=np.linspace(0, 1, 20), color='k')
    axs[0].set_xlabel('OSI')

    axs[1].hist(dsis, bins=np.linspace(0, 1, 20), color='k')
    axs[1].set_xlabel('DSI')

    axs[2].hist(rel['r'], bins=np.linspace(-1, 1, 40), color='k')
    axs[2].set_xlabel('odd/even tuning r')

    fig.tight_layout()
    fig.savefig(os.path.join(gt_dir, 'gratings_selectivities.png'))

    n_plot_cells = min(100, sps.shape[0])
    with PdfPages(os.path.join(gt_dir, 'gratings_first_{}_cells.pdf'.format(n_plot_cells))) as pdf:
        fig, axs = plt.subplots(10, 10, figsize=(8.5, 11))
        axs = axs.flatten()

        for c in range(n_plot_cells):
            axs[c].plot(orientations, gratings_data[c,:,0], color='k', lw=0.75)
            axs[c].fill_between(
                orientations,
                gratings_data[c,:,0]-gratings_data[c,:,1],
                gratings_data[c,:,0]+gratings_data[c,:,1],
                alpha=0.5, color='k', lw=0
            )
            axs[c].set_xticks([0, 90, 180, 270])
            axs[c].tick_params(labelsize=4, length=1.5, pad=1)
            axs[c].set_ylim([
                0,
                np.nanmax(gratings_data[c,:,0]+gratings_data[c,:,1])
            ])
            axs[c].set_title('cell {}\nosi={:.2f}, dsi={:.2f}, r={:.2f}'.format(
                c, osis[c], dsis[c], rel['r'][c]), fontsize=5, pad=2)
        for ax in axs[n_plot_cells:]:
            ax.axis('off')

        fig.suptitle('Direction tuning, first {} cells (x: direction, deg; y: {})'.format(
            n_plot_cells, LABELS[activity_source]), fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=0.6, w_pad=0.3)
        pdf.savefig(fig)
        plt.close(fig)

    return results

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    parser.add_argument('--time_path', default=None,
                        help='Gratings timestamp csv, if it is not inside --path')
    parser.add_argument('--n_shuffles', type=int, default=1000)
    parser.add_argument('--activity', choices=SOURCES, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    if args.path is None:
        usedir = select_directory('Select gratings directory')
    else:
        usedir = args.path
    
    main(usedir, time_path=args.time_path, n_shuffles=args.n_shuffles, activity_source=args.activity)

