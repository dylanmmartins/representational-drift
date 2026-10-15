import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os

from files import find
from twop_helpers import read_xml
from gui_funcs import select_directory


def main(gt_dir):

    time_path = find(path=gt_dir, pattern='gratings_timestamps*.csv', MR=True)
    gt_time = pd.read_csv(time_path)

    xml_data = read_xml(find(path=gt_dir, pattern='*.xml', MR=True))

    sps = np.load(os.path.join(gt_dir, 'suite2p/plane0/spks.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(gt_dir, 'suite2p/plane0/iscell.npy'), allow_pickle=True)
    sps = sps[iscell[:,0].astype(bool)] # keep only ROIs classified as cells
    twopT = xml_data['rel_time']

    orientations = np.arange(0, 360, 30)
    trial_oris = gt_time['orientation_deg'].astype(int).values

    missing = set(orientations) - set(trial_oris)
    if missing:
        print('Warning: no presentations of orientations {}'.format(sorted(missing)))
    unexpected = set(trial_oris) - set(orientations)
    if unexpected:
        print('Warning: presented orientations {} not in analysis orientations'.format(sorted(unexpected)))

    # mean activity of each cell during each trial
    trial_resp = np.zeros([sps.shape[0], len(gt_time)]) * np.nan
    for i in range(len(gt_time)):
        start = gt_time['onset_s'][i]
        end = gt_time['offset_s'][i]
        spike_indices = np.where((twopT >= start) & (twopT <= end))[0]
        if len(spike_indices) > 0:
            trial_resp[:,i] = np.nanmean(sps[:, spike_indices], axis=1)

    # index by orientation so a missing orientation stays NaN instead of
    # shifting the following orientations into the wrong columns
    gratings_data = np.zeros([sps.shape[0], len(orientations), 2]) * np.nan
    for o, ori in enumerate(orientations):
        frs = trial_resp[:, trial_oris==ori]
        if frs.shape[1] > 0:
            gratings_data[:,o,0] = np.nanmean(frs, axis=1)
            gratings_data[:,o,1] = np.nanstd(frs, axis=1) / np.sqrt(frs.shape[1])

    # orientation and direction selectivity index
    def calc_gratings_selectivity(tuning):
        n_dirs = len(tuning) # directions evenly spaced around 360 deg
        half_turn = n_dirs // 2 # index step for 180 deg
        quarter_turn = n_dirs // 4 # index step for 90 deg

        th_pref = np.nanargmax(tuning, axis=0)
        th_null = (th_pref + half_turn) % n_dirs # other direction of same orientation
        th_ortho = (th_pref + quarter_turn) % n_dirs # orthogonal orientation

        R_pref_ori = (tuning[th_pref] + tuning[th_null]) * 0.5 # preferred orientation (avg of both directions)
        R_ortho = (tuning[th_ortho] + tuning[(th_ortho + half_turn) % n_dirs]) * 0.5 # orthogonal orientation (avg of both directions)
        R_pref_dir = tuning[th_pref] # preferred direction
        R_null = tuning[th_null] # opposite direction

        osi = (R_pref_ori - R_ortho) / (R_pref_ori + R_ortho)
        dsi = (R_pref_dir - R_null) / (R_pref_dir + R_null)

        return osi, dsi


    osis = np.zeros(sps.shape[0]) * np.nan
    dsis = np.zeros(sps.shape[0]) * np.nan

    for c in range(sps.shape[0]):
        osis[c], dsis[c] = calc_gratings_selectivity(gratings_data[c, :, 0])


    fig, axs = plt.subplots(1, 2, figsize=(6,2), dpi=300)
    axs[0].hist(osis, bins=np.linspace(0, 1, 20), color='k')
    axs[0].set_xlabel('OSI')

    axs[1].hist(dsis, bins=np.linspace(0, 1, 20), color='k')
    axs[1].set_xlabel('DSI')

    fig.tight_layout()
    fig.savefig('gratings_selectivities.png')

    # tuning curves of the first 100 cells on a letter-size page
    n_plot_cells = min(100, sps.shape[0])
    with PdfPages('gratings_first_{}_cells.pdf'.format(n_plot_cells)) as pdf:
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
            axs[c].set_title('cell {}\nosi={:.2f}, dsi={:.2f}'.format(c, osis[c], dsis[c]), fontsize=5, pad=2)
        for ax in axs[n_plot_cells:]:
            ax.axis('off')

        fig.suptitle('Direction tuning, first {} cells (x: direction, deg; y: mean spks)'.format(
            n_plot_cells), fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=0.6, w_pad=0.3)
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    args = parser.parse_args()

    if args.path is None:
        usedir = select_directory('Select gratings directory')
    else:
        usedir = args.path
    
    main(usedir)