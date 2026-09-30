import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from utils.files import find
from twop_helpers import read_xml

gt_dir = '/home/dylan/Fast2/repdrift/week1/260929_DMM_DMM085_week1/pg'

time_path = find(path=gt_dir, pattern='penguin_time*.csv', MR=True)
mv_time = pd.read_csv(time_path)

xml_data = read_xml(find(path=gt_dir, pattern='*.xml', MR=True))

sps = np.load(os.path.join(gt_dir, 'suite2p/plane0/spks.npy'), allow_pickle=True)
iscell = np.load(os.path.join(gt_dir, 'suite2p/plane0/iscell.npy'), allow_pickle=True)
sps = sps[iscell[:,0].astype(bool)] # keep only ROIs classified as cells
twopT = xml_data['rel_time']

mv_time = mv_time.sort_values(['repeat', 'movie_frame']).reset_index(drop=True)
nRep = mv_time['repeat'].nunique()
nF = mv_time['movie_frame'].nunique()
nCells = sps.shape[0]
if len(mv_time) != nRep * nF:
    raise ValueError('Expected {} repeats x {} frames = {} rows in {}, found {}'.format(
        nRep, nF, nRep * nF, time_path, len(mv_time)))

movie_onsets = mv_time['onset_s'].values.reshape(nRep, nF)

def get_movie_frame_indices(lag):
    # index of the 2P frame being acquired at each movie-frame onset, with movie times
    # shifted by lag (s). The movie (30 Hz) is faster than imaging (10 Hz), so ~3
    # neighbouring movie frames share the same 2P frame.
    return np.searchsorted(twopT, movie_onsets + lag, side='right') - 1

# Offset (s) between the stimulus clock (psychopy) and 2P clock, which are not
# synchronized. Unlike the gratings, it cannot be estimated with a split-half lag
# scan: the movie repeats back-to-back, so shifting the stimulus times shifts the
# even and odd repeats together and their split-half reliability stays the same
# at every lag. Set this from a sync signal (photodiode/TTL) when available.
lag = 0.
print('Warning: movie stimulus times use lag = {:.2f} s; the stimulus/2P offset cannot be '
      'estimated from looped movie data, so responses may be misaligned in time.'.format(lag))

resp = sps[:, get_movie_frame_indices(lag)] # (nCells, nRep, nF)

# Visualize the movie responses of the first few cells: every repeat as a heatmap
# (top) and the mean across repeats +/- SEM (bottom).
n_plot_cells = 8
movie_time = movie_onsets[0] - movie_onsets[0, 0] # (s) from the start of the movie

with PdfPages('movie_first_{}_cells.pdf'.format(n_plot_cells)) as pdf:
    fig, axs = plt.subplots(n_plot_cells, 2, figsize=(8.5, 11), sharex=True)
    for c in range(min(n_plot_cells, nCells)):
        axs[c,0].imshow(
            resp[c], aspect='auto', cmap='gray_r', interpolation='nearest',
            extent=[movie_time[0], movie_time[-1], nRep, 0]
        )
        axs[c,0].set_ylabel('cell {}\nrepeat'.format(c), fontsize=7)

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
    fig.suptitle('Movie responses (lag = {:.2f} s)'.format(lag), fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
