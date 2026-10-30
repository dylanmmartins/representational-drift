import os
import re
import sys
import glob
import warnings
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import find_contours

sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
from twop_helpers import read_xml
from activity import load_activity, frame_rate
from gui_funcs import select_directory


def main(week1_path, week2_path):

    print('Recording subdirectories need to match these names exactly (order does not matter): antelopes meerkatspenguins gratings sparse_noise')

    DAY_DIRS = {
        'week 1': week1_path,
        'week 2': week2_path,
    }
    REG_PATH = os.path.join(DAY_DIRS['week 2'], 'all_tifs__all_tifs_reg.csv')
    PLANE = 'all_tifs/suite2p/plane0'
    if 'suite2p' in week2_path:
        OUT_DIR = os.path.split(os.path.split(week2_path)[0])[0]
    elif 'suite2p' not in week2_path and 'plane0' not in week2_path:
        OUT_DIR = week2_path
    else:
        print('Cannot figure out the save path, exiting.')
        return

    NULL_PCTL = 99
    STA_SIGMA = 2.0
    N_SN_NULL = 5
    CROP = 18
    PATCH_WIN, PATCH_STEP = 64, 32
    N_CANDIDATES = 10
    NEAR_MISS_PX = 4.0
    OFFSET_SWEEP = np.arange(-4, 25, 0.1)
    WRITE_ALL_CELLS_PDF = True
    SEED = 0

    DAYS = list(DAY_DIRS)
    MOVIES = ['antelopes', 'meerkats', 'penguins']
    STIMS = ['gratings'] + MOVIES + ['sparse_noise']
    LABEL = {'gratings': 'Gratings', 'antelopes': 'Antelopes', 'meerkats': 'Meerkats',
            'penguins': 'Penguins', 'sparse_noise': 'Sparse noise'}
    TS_PREFIX = {'gratings': 'gratings', 'antelopes': 'antelope', 'meerkats': 'meerkat',
                'penguins': 'penguin', 'sparse_noise': 'sparse_noise'}

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    COLOR = {'gratings': '#2a78d6', 'antelopes': '#eb6834', 'meerkats': '#1baf7a',
            'penguins': '#eda100', 'sparse_noise': '#e87ba4'}
    INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
    GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
    DAY_COLOR = {'week 1': INK2}
    IMG_COLOR = {'week 1': '#ff3df5', 'week 2': '#3dff6e'}
    DIVERGING = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a99', '#b02a2a'])
    SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#e6eefa', '#86b6ef', '#2a78d6', '#123f78'])

    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.family': 'sans-serif', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.5, 'lines.linewidth': 1.5, 'legend.frameon': False,
        'legend.fontsize': 7, 'text.color': INK, 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
    })


    def savefig(fig, name):
        fig.savefig(os.path.join(OUT_DIR, name + '.png'), bbox_inches='tight')
        # fig.savefig(os.path.join(OUT_DIR, name + '.pdf'), bbox_inches='tight')


    def no_ticks(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)


    def ecdf(ax, v, **kw):
        v = np.sort(v[np.isfinite(v)])
        ax.plot(v, np.arange(1, len(v) + 1) / max(len(v), 1), **kw)


    reg = pd.read_csv(REG_PATH)
    reg_npz = np.load(REG_PATH.replace('.csv', '.npz'))
    for d, key in zip(DAYS, ['day1_path', 'day2_path']):
        assert str(reg_npz[key]).startswith(DAY_DIRS[d])
    A = reg_npz['A_moving_to_fixed']
    A_inv = np.linalg.inv(A)
    print('transform:', reg_npz['transform_source'])
    print(np.round(A, 4))

    matched = reg[reg.status == 'matched'].reset_index(drop=True)
    N = len(matched)
    if N == 0:
        raise SystemExit('{} has no matched ROI pairs. Run "Detect overlap" in the registration GUI '
                         'before saving (changing the transform clears existing matches).'.format(REG_PATH))
    raw = {DAYS[0]: matched.day1_raw.astype(int).to_numpy(), DAYS[1]: matched.day2_raw.astype(int).to_numpy()}
    idx = {DAYS[0]: matched.day1_iscell.astype(int).to_numpy(), DAYS[1]: matched.day2_iscell.astype(int).to_numpy()}

    s2p = {}
    for d in DAYS:
        pdir = os.path.join(DAY_DIRS[d], PLANE)
        ops = np.load(os.path.join(pdir, 'ops.npy'), allow_pickle=True).item()
        stat = np.load(os.path.join(pdir, 'stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(pdir, 'iscell.npy'))[:, 0] == 1
        cen = np.array([[np.average(s['xpix'], weights=s['lam']), np.average(s['ypix'], weights=s['lam'])] for s in stat])
        s2p[d] = dict(ops=ops, stat=stat, iscell_raw=np.where(iscell)[0], cen=cen,
                    npix=np.array([s['npix'] for s in stat]), shape=(ops['Ly'], ops['Lx']))
        assert np.array_equal(s2p[d]['iscell_raw'][idx[d]], raw[d]), 'iscell indices in the csv do not match iscell.npy'


    def w2_to_w1(xy):
        return np.asarray(xy, float) @ A[:2, :2].T + A[:2, 2]


    def w1_to_w2(xy):
        return np.asarray(xy, float) @ A_inv[:2, :2].T + A_inv[:2, 2]


    def sample(img, xy, order=1):
        xy = np.asarray(xy, float)
        return map_coordinates(img.astype(float), [xy[..., 1], xy[..., 0]], order=order, mode='constant', cval=np.nan)


    def pixel_grid(shape):
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        return np.stack([xx, yy], axis=-1).astype(float)


    def warp_w2(img2):
        return sample(img2, w1_to_w2(pixel_grid(s2p[DAYS[0]]['shape'])))


    c1 = s2p[DAYS[0]]['cen'][raw[DAYS[0]]]
    c2 = s2p[DAYS[1]]['cen'][raw[DAYS[1]]]
    c2_in_1 = w2_to_w1(c2)
    resid = c1 - c2_in_1
    resid_d = np.linalg.norm(resid, axis=1)
    check = pd.Series({
        'raw (no transform)': np.median(np.linalg.norm(c1 - c2, axis=1)),
        'A (week 2 -> week 1)': np.median(resid_d),
        'inverse of A': np.median(np.linalg.norm(c1 - (c2 @ A_inv[:2, :2].T + A_inv[:2, 2]), axis=1)),
    }, name='median centroid distance of matched pairs (px)')

    def norm01(img, lo=1, hi=99.5):
        a, b = np.nanpercentile(img, [lo, hi])
        return np.clip((img - a) / (b - a), 0, 1)


    def composite(i1, i2):
        a, b = np.nan_to_num(norm01(i1)), np.nan_to_num(norm01(i2))
        return np.stack([a, b, a], axis=-1)


    def nancorr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 10 else np.nan


    img1 = s2p[DAYS[0]]['ops']['meanImg']
    img2 = s2p[DAYS[1]]['ops']['meanImg']
    img2_w = warp_w2(img2)
    ys, xs = s2p[DAYS[0]]['ops']['yrange'], s2p[DAYS[0]]['ops']['xrange']
    inner = np.zeros(img1.shape, bool); inner[ys[0]:ys[1], xs[0]:xs[1]] = True
    r_raw = nancorr(np.where(inner, img1, np.nan), img2)
    r_reg = nancorr(np.where(inner, img1, np.nan), img2_w)

    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, im, title in zip(axs[:2], [img1, img2], ['{} meanImg'.format(DAYS[0]), '{} meanImg'.format(DAYS[1])]):
        ax.imshow(norm01(im), cmap='gray'); no_ticks(ax); ax.set_title(title)
    axs[2].imshow(composite(img1, img2)); no_ticks(axs[2])
    axs[2].set_title('Overlay, no transform  (r = {:.2f})'.format(r_raw))
    axs[3].imshow(composite(img1, img2_w)); no_ticks(axs[3])
    axs[3].set_title('Overlay, week 2 warped  (r = {:.2f})'.format(r_reg))
    fig.tight_layout()
    savefig(fig, 'diag1_fov_overlay')

    def local_corr(i1, i2, win=PATCH_WIN, step=PATCH_STEP):
        H, W = i1.shape
        starts_y, starts_x = range(0, H - win + 1, step), range(0, W - win + 1, step)
        out = np.full((len(starts_y), len(starts_x)), np.nan)
        for a, y0 in enumerate(starts_y):
            for b, x0 in enumerate(starts_x):
                p1, p2 = i1[y0:y0 + win, x0:x0 + win], i2[y0:y0 + win, x0:x0 + win]
                if np.mean(np.isfinite(p1) & np.isfinite(p2)) > 0.8:
                    out[a, b] = nancorr(p1, p2)
        return out, (win / 2 - 0.5, step)


    i1_inner = np.where(inner, img1, np.nan)
    lc_raw, _ = local_corr(i1_inner, img2)
    lc_reg, (off, step) = local_corr(i1_inner, img2_w)
    H, W = img1.shape
    extent = [off - step / 2, off + step * (lc_reg.shape[1] - 0.5), off + step * (lc_reg.shape[0] - 0.5), off - step / 2]

    def roi_contour(st):
        y, x = st['ypix'], st['xpix']
        y0, x0 = y.min() - 1, x.min() - 1
        m = np.zeros((y.max() - y0 + 2, x.max() - x0 + 2))
        m[y - y0, x - x0] = 1
        c = max(find_contours(m, 0.5), key=len)
        return np.stack([c[:, 1] + x0, c[:, 0] + y0], axis=1)


    contours = {d: {r: roi_contour(s2p[d]['stat'][r]) for r in s2p[d]['iscell_raw']} for d in DAYS}
    unmatched = {DAYS[0]: reg.day1_raw[reg.status == 'day1_only'].astype(int).to_numpy(),
                DAYS[1]: reg.day2_raw[reg.status == 'day2_only'].astype(int).to_numpy()}


    def draw_rois(ax):
        ax.imshow(norm01(img1) * 0.55, cmap='gray', vmin=0, vmax=1)
        for d in DAYS:
            for r in unmatched[d]:
                c = contours[d][r] if d == DAYS[0] else w2_to_w1(contours[d][r])
                ax.plot(c[:, 0], c[:, 1], color=AXIS, lw=0.3, alpha=0.6)
        for k in range(N):
            a = contours[DAYS[0]][raw[DAYS[0]][k]]
            b = w2_to_w1(contours[DAYS[1]][raw[DAYS[1]][k]])
            ax.plot(a[:, 0], a[:, 1], color=IMG_COLOR['week 1'], lw=0.6)
            ax.plot(b[:, 0], b[:, 1], color=IMG_COLOR['week 2'], lw=0.6)
        no_ticks(ax)


    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    draw_rois(axs[0])
    axs[0].set_title('Matched ROIs: week 1 magenta, week 2 green (grey: unmatched)')
    draw_rois(axs[1])
    # axs[1].set_xlim(cx - 100, cx + 100); axs[1].set_ylim(cy + 100, cy - 100)
    axs[1].set_title('Center 200x200 px')
    fig.tight_layout()
    savefig(fig, 'diag3_roi_contours')


    tree = {DAYS[0]: cKDTree(s2p[DAYS[0]]['cen'][s2p[DAYS[0]]['iscell_raw']]),
            DAYS[1]: cKDTree(w2_to_w1(s2p[DAYS[1]]['cen'][s2p[DAYS[1]]['iscell_raw']]))}
    miss1 = tree[DAYS[1]].query(s2p[DAYS[0]]['cen'][unmatched[DAYS[0]]])[0]
    miss2 = tree[DAYS[0]].query(w2_to_w1(s2p[DAYS[1]]['cen'][unmatched[DAYS[1]]]))[0]
    area_ratio = s2p[DAYS[1]]['npix'][raw[DAYS[1]]] * abs(np.linalg.det(A[:2, :2])) / s2p[DAYS[0]]['npix'][raw[DAYS[0]]]

    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), gridspec_kw={'width_ratios': [1.15, 1, 1, 1]})
    ax = axs[0]
    ax.imshow(norm01(img1) * 0.4, cmap='gray', vmin=0, vmax=1)
    ax.quiver(c2_in_1[:, 0], c2_in_1[:, 1], resid[:, 0], resid[:, 1], angles='xy', scale_units='xy', scale=1 / 20,
            color=COLOR['gratings'], width=0.004, headwidth=3)
    no_ticks(ax)
    ax.set_title('Residuals (x20), mean ({:+.2f}, {:+.2f}) px'.format(*resid.mean(axis=0)))

    ax = axs[1]
    bins = np.linspace(0, 12, 25)
    ax.hist(resid_d, bins=bins, color=COLOR['gratings'], rwidth=0.9, label='matched pairs')
    ax.set_xlabel('distance (px)'); ax.set_ylabel('pairs')
    ax.set_title('Residual: median {:.1f} px'.format(np.median(resid_d)))

    ax = axs[2]
    bins = np.linspace(0, 40, 41)
    ax.hist(np.concatenate([miss1, miss2]), bins=bins, color=AXIS, rwidth=0.9, label='unmatched ROI to nearest\nROI on the other day')
    ax.hist(resid_d, bins=bins, histtype='step', color=COLOR['gratings'], lw=1.5, label='matched pairs')
    ax.set_xlabel('distance (px)'); ax.set_ylabel('ROIs')
    ax.legend(loc='upper right')
    ax.set_title('Near misses')

    ax = axs[3]
    ax.scatter(resid_d, matched.p_overlap, s=5, color=COLOR['gratings'], lw=0, alpha=0.6)
    ax.set_xlabel('residual (px)'); ax.set_ylabel('ROI overlap')
    rho = stats.spearmanr(resid_d, matched.p_overlap)[0]
    ax.set_title('Overlap vs residual (rho = {:.2f})'.format(rho))
    fig.tight_layout()
    savefig(fig, 'diag4_centroid_residuals')

    X = np.column_stack([np.ones(N), c1])
    coef = np.linalg.lstsq(X, resid, rcond=None)[0]
    print('residual vs position: d(residual)/dx = ({:+.4f}, {:+.4f}), d/dy = ({:+.4f}, {:+.4f})  px per px'.format(
        *coef[1], *coef[2]))
    print('i.e. a {:.2f} px change in residual across the {} px field of view'.format(
        np.abs(coef[1:]).max() * img1.shape[0], img1.shape[0]))
    print('unmatched ROIs within 4 px of an ROI on the other day: {} of {} ({:.1%})'.format(
        np.sum(np.concatenate([miss1, miss2]) < 4), len(miss1) + len(miss2), np.mean(np.concatenate([miss1, miss2]) < 4)))
    print('area ratio (week 2 / week 1, scale-corrected): median {:.2f}, IQR {:.2f}-{:.2f}'.format(
        np.median(area_ratio), *np.percentile(area_ratio, [25, 75])))

    def crop_points(c, h=CROP):
        off = np.arange(-h, h + 1, dtype=float)
        gx, gy = np.meshgrid(c[0] + off, c[1] + off)
        return np.stack([gx, gy], axis=-1)


    def footprint_img(d, r):
        st = s2p[d]['stat'][r]
        z = np.zeros(s2p[d]['shape'])
        z[st['ypix'], st['xpix']] = st['lam']
        return z


    def patch_w1(img, c):
        return sample(img, crop_points(c))


    def patch_w2_in_w1(img, c):
        return sample(img, w1_to_w2(crop_points(c)))


    perm = np.array([rng.choice(np.delete(np.arange(N), k)) for k in range(N)])
    patch_r = np.array([nancorr(patch_w1(img1, c1[k]), patch_w2_in_w1(img2, c1[k])) for k in range(N)])
    patch_r_raw = np.array([nancorr(patch_w1(img1, c1[k]), patch_w1(img2, c1[k])) for k in range(N)])
    patch_r_null = np.array([nancorr(patch_w1(img1, c1[k]), patch_w2_in_w1(img2, c1[perm[k]])) for k in range(N)])

    fp1 = [patch_w1(footprint_img(DAYS[0], raw[DAYS[0]][k]), c1[k]) for k in range(N)]
    fp2 = [patch_w2_in_w1(footprint_img(DAYS[1], raw[DAYS[1]][k]), c2_in_1[k]) for k in range(N)]
    shape_r = np.array([nancorr(fp1[k], fp2[k]) for k in range(N)])
    shape_r_null = np.array([nancorr(fp1[k], fp2[perm[k]]) for k in range(N)])

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    ax = axs[0]
    ecdf(ax, patch_r, color=COLOR['gratings'], label='matched, week 2 warped')
    ecdf(ax, patch_r_raw, color=INK2, ls='--', label='matched, no transform')
    ecdf(ax, patch_r_null, color=MUTED, label='different cell')
    ax.set_xlim(-0.5, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('mean-image patch r'); ax.set_ylabel('fraction of pairs')
    ax.legend(loc='upper left'); ax.set_title('Anatomy around the cell')
    ax = axs[1]
    ecdf(ax, shape_r, color=COLOR['gratings'], label='matched')
    ecdf(ax, shape_r_null, color=MUTED, label='random pair')
    ax.set_xlim(-0.2, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('centered footprint r'); ax.set_ylabel('fraction of pairs')
    ax.legend(loc='upper left'); ax.set_title('ROI shape')
    ax = axs[2]
    ax.hist(np.log2(area_ratio), bins=np.linspace(-2, 2, 33), color=COLOR['gratings'], rwidth=0.9)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_xticklabels(['1/4', '1/2', '1', '2', '4'])
    ax.set_xlabel('ROI area, week 2 / week 1'); ax.set_ylabel('pairs')
    ax.set_title('ROI size')
    fig.tight_layout()
    savefig(fig, 'diag5_anatomy_morphology')

    pd.DataFrame({
        'median': [np.nanmedian(v) for v in (patch_r, patch_r_raw, patch_r_null, shape_r, shape_r_null)],
    }, index=['patch r, warped', 'patch r, no transform', 'patch r, different cell', 'shape r, matched', 'shape r, random pair']).round(3)


    rates = {d: {} for d in DAYS}
    for d in DAYS:
        for s in STIMS:
            z = np.load(os.path.join(DAY_DIRS[d], s, 'suite2p/plane0/omsi_spikes.npz'))
            rates[d][s] = z['prob_trace'].mean(axis=1) * float(z['hz'])

    floor = 1e-3
    fig, axs = plt.subplots(1, len(STIMS), figsize=(11, 2.4))
    silent_rows = []
    for ax, s in zip(axs, STIMS):
        a, b = rates[DAYS[0]][s][idx[DAYS[0]]], rates[DAYS[1]][s][idx[DAYS[1]]]
        ax.plot([floor, 10], [floor, 10], color=AXIS, lw=0.8)
        ax.scatter(np.maximum(a, floor), np.maximum(b, floor), s=4, color=COLOR[s], lw=0, alpha=0.6)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(floor * 0.7, 10); ax.set_ylim(floor * 0.7, 10); ax.set_aspect('equal')
        ax.set_title('{}  (r = {:.2f})'.format(LABEL[s], stats.spearmanr(a, b)[0]))
        ax.set_xlabel('spikes/s, {}'.format(DAYS[0]))
        silent_rows.append({'stimulus': LABEL[s], 'silent on {} only'.format(DAYS[0]): int(np.sum((a == 0) & (b > 0))),
                            'silent on {} only'.format(DAYS[1]): int(np.sum((b == 0) & (a > 0))),
                            'silent both': int(np.sum((a == 0) & (b == 0))),
                            'median rate ratio (wk2/wk1)': np.nanmedian(b[(a > 0) & (b > 0)] / a[(a > 0) & (b > 0)])})
    axs[0].set_ylabel('spikes/s, {}'.format(DAYS[1]))
    fig.suptitle('Mean spike rate of matched cells (silent cells pinned to the axis floor)', x=0.01, ha='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'diag6_rates')

    def rec_dir(d, s):
        return os.path.join(DAY_DIRS[d], s)


    def find_timestamps(d, s):
        pat = '{}_timestamps*.csv'.format(TS_PREFIX[s])
        hits = glob.glob(os.path.join(rec_dir(d, s), pat)) or glob.glob(os.path.join(DAY_DIRS[d], pat))
        return max(hits, key=os.path.getmtime)


    def find_xml(d, s):
        xmls = [x for x in glob.glob(os.path.join(rec_dir(d, s), '*.xml')) if 'Voltage' not in os.path.basename(x)]
        return max(xmls, key=os.path.getmtime)


    timing_rows = []
    for d in DAYS:
        for s in STIMS:
            with open(find_xml(d, s), 'rb') as f:
                head = f.read(300000).decode('utf-8', 'ignore')
            seq = re.search(r'<Sequence [^>]*time="([\d:.]+)"', head).group(1)
            first = float(re.search(r'<Frame relativeTime="[^"]*" absoluteTime="([\d.]+)"', head).group(1))
            ts_path = find_timestamps(d, s)
            stamp = re.search(r'_(\d{8}_\d{6})\.csv$', ts_path).group(1)
            t_script = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
            hms, frac = seq.split('.')
            t_armed = datetime.strptime('{} {}.{}'.format(stamp[:8], hms, frac[:6]), '%Y%m%d %H:%M:%S.%f')
            stim = pd.read_csv(ts_path)
            twopT = read_xml(find_xml(d, s))['rel_time']
            timing_rows.append({
                'day': d, 'stimulus': LABEL[s], 'script started': t_script.strftime('%H:%M:%S'),
                'first 2p frame': (t_armed + pd.Timedelta(seconds=first)).strftime('%H:%M:%S.%f')[:-4],
                'first frame - script start (s)': (t_armed - t_script).total_seconds() + first,
                'imaging (s)': twopT[-1], 'stimulus end (s)': stim['offset_s'].iloc[-1],
                'imaging after stimulus end (s)': twopT[-1] - stim['offset_s'].iloc[-1],
            })
    timing = pd.DataFrame(timing_rows).set_index(['day', 'stimulus'])


    movie_src = {d: {} for d in DAYS}
    for d in DAYS:
        for m in MOVIES:
            twopT = read_xml(find_xml(d, m))['rel_time']
            act = load_activity(os.path.join(rec_dir(d, m), 'suite2p/plane0'), frame_rate(twopT))
            mt = pd.read_csv(find_timestamps(d, m)).sort_values(['repeat', 'movie_frame']).reset_index(drop=True)
            n_rep, n_f = mt['repeat'].nunique(), mt['movie_frame'].nunique()
            movie_src[d][m] = dict(twopT=twopT, act=np.asarray(act, np.float32),
                                onsets=mt['onset_s'].to_numpy().reshape(n_rep, n_f))


    def movie_resp(d, m, offset=0.0, rows=None):

        src = movie_src[d][m]
        k = np.clip(np.searchsorted(src['twopT'], src['onsets'] + offset, side='right') - 1, 0, len(src['twopT']) - 1)
        a = src['act'] if rows is None else src['act'][rows]
        return a[:, k]


    for d in DAYS:
        for m in MOVIES:
            with h5py.File(os.path.join(rec_dir(d, m), 'movie.h5'), 'r') as f:
                ref = f['resp'][:50]
            assert np.allclose(movie_resp(d, m, 0.0, np.arange(50)), ref, rtol=1e-4, atol=1e-6), (d, m)


    def rowcorr(a, b):
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
        r = np.nansum(a * b, axis=1) / den
        r[~(den > 0)] = np.nan
        return r


    sweep = {}
    for m in MOVIES:
        n_f = min(movie_src[d][m]['onsets'].shape[1] for d in DAYS)
        ref = movie_resp(DAYS[1], m, 0.0, idx[DAYS[1]]).mean(axis=1)[:, :n_f]
        src = movie_src[DAYS[0]][m]
        max_off = src['twopT'][-1] - src['onsets'][-1, -1]
        offs = OFFSET_SWEEP[OFFSET_SWEEP <= max_off]
        curve = np.array([np.nanmedian(rowcorr(movie_resp(DAYS[0], m, o, idx[DAYS[0]]).mean(axis=1)[:, :n_f], ref))
                        for o in offs])
        pos = offs >= 0
        o0 = offs[pos][np.argmax(curve[pos])]
        fine = np.arange(max(o0 - 0.3, 0), min(o0 + 0.3, max_off), 0.01)
        fcurve = np.array([np.nanmedian(rowcorr(movie_resp(DAYS[0], m, o, idx[DAYS[0]]).mean(axis=1)[:, :n_f], ref))
                        for o in fine])
        sweep[m] = dict(offs=offs, curve=curve, best=fine[np.argmax(fcurve)], best_r=fcurve.max(),
                        r0=np.nanmedian(rowcorr(movie_resp(DAYS[0], m, 0.0, idx[DAYS[0]]).mean(axis=1)[:, :n_f], ref)),
                        loop=np.median(np.diff(src['onsets'][:, 0])))

    OFFSET = {(d, m): 0.0 for d in DAYS for m in MOVIES}
    for m in MOVIES:
        if sweep[m]['best_r'] - sweep[m]['r0'] > 0.05:
            OFFSET[(DAYS[0], m)] = round(float(sweep[m]['best']), 2)


    fig, axs = plt.subplots(1, 4, figsize=(11, 2.7), gridspec_kw={'width_ratios': [1.4, 1, 1, 1]})
    ax = axs[0]
    for m in MOVIES:
        ax.plot(sweep[m]['offs'], sweep[m]['curve'], color=COLOR[m], lw=1.5, label=LABEL[m])
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.annotate('{:+.2f} s'.format(sweep['antelopes']['best']), (sweep['antelopes']['best'], sweep['antelopes']['best_r']),
                xytext=(4, 2), textcoords='offset points', fontsize=7, color=INK2)
    ax.set_xlabel('assumed delay of week 1 movie onset (s)')
    ax.set_ylabel('median cross-day r')
    ax.legend(loc='upper left')
    ax.set_title('Cross-day sweep (week 2 at logged timing)')

    win_s, smooth_s = 30, 0.5
    for ax, m in zip(axs[1:], MOVIES):
        for d, color in zip(DAYS, [DAY_COLOR['week 1'], COLOR[m]]):
            src = movie_src[d][m]
            fr = frame_rate(src['twopT'])
            n = int(win_s * fr)
            pop = src['act'][idx[d]].mean(axis=0)
            k = max(int(round(smooth_s * fr)), 1)
            pop = np.convolve(pop, np.ones(k) / k, mode='same')
            ax.plot(src['twopT'][:n], pop[:n], color=color, lw=1.2, label=d)
        ax.axvline(0, color=AXIS, lw=0.8)
        if OFFSET[(DAYS[0], m)]:
            ax.axvline(OFFSET[(DAYS[0], m)], color=INK2, lw=0.8, ls='--')
            ax.text(OFFSET[(DAYS[0], m)], ax.get_ylim()[0], ' corrected week 1 onset', va='bottom', fontsize=6.5, color=INK2)
        ax.set_xlabel('imaging time (s)')
        ax.set_title('{}: start of recording'.format(LABEL[m]))
    axs[1].set_ylabel('mean spikes/s (matched cells)')
    axs[1].legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, 'diag7_antelope_timing')

    def movie_reliability(R):
        return rowcorr(R[:, 0::2].mean(axis=1), R[:, 1::2].mean(axis=1))


    rel_movie, thresh_movie = {d: {} for d in DAYS}, {d: {} for d in DAYS}
    for d in DAYS:
        for m in MOVIES:
            with h5py.File(os.path.join(rec_dir(d, m), 'movie.h5'), 'r') as f:
                rel_h5 = f['rel_r'][:]
                thresh_movie[d][m] = np.nanpercentile(f['rel_null'][:], NULL_PCTL, axis=1)
            rel_movie[d][m] = movie_reliability(movie_resp(d, m, OFFSET[(d, m)]))
            if OFFSET[(d, m)] == 0:
                assert np.allclose(np.nan_to_num(rel_movie[d][m]), np.nan_to_num(rel_h5), atol=1e-4)

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    rows = []
    for m in MOVIES:
        n_f = min(movie_src[d][m]['onsets'].shape[1] for d in DAYS)
        B = movie_resp(DAYS[1], m, 0.0, idx[DAYS[1]]).mean(axis=1)[:, :n_f]
        rel_ok = rel_movie[DAYS[0]][m][idx[DAYS[0]]] > thresh_movie[DAYS[0]][m][idx[DAYS[0]]]
        for off, ls in ([(0.0, '--'), (OFFSET[(DAYS[0], m)], '-')] if OFFSET[(DAYS[0], m)] else [(0.0, '-')]):
            r = rowcorr(movie_resp(DAYS[0], m, off, idx[DAYS[0]]).mean(axis=1)[:, :n_f], B)[rel_ok]
            lab = '{}{}'.format(LABEL[m], '' if not OFFSET[(DAYS[0], m)] else (' (logged timing)' if off == 0 else ' (corrected)'))
            ecdf(ax, r, color=COLOR[m], ls=ls, label=lab)
            wk1 = movie_reliability(movie_resp(DAYS[0], m, off, idx[DAYS[0]]))
            rows.append({'movie': lab, 'n (reliable on week 1)': int(rel_ok.sum()), 'median cross-day r': np.nanmedian(r),
                        'median week 1 reliability': np.nanmedian(wk1[rel_ok])})
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlim(-0.5, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('cross-day r of trial-averaged response'); ax.set_ylabel('fraction of cells')
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_title('Movies: same cell, week 1 vs week 2')
    fig.tight_layout()
    savefig(fig, 'diag8_antelope_correction')
    pd.DataFrame(rows).set_index('movie').round(3)


    def smooth_sta(sta, shape):
        return gaussian_filter(sta.reshape(-1, *shape), sigma=(0, STA_SIGMA, STA_SIGMA)).reshape(len(sta), -1)


    def derangement(n):
        while True:
            p = rng.permutation(n)
            if not np.any(p == np.arange(n)):
                return p


    func = {d: {} for d in DAYS}
    for d in DAYS:
        with h5py.File(os.path.join(rec_dir(d, 'gratings'), 'gratings.h5'), 'r') as f:
            func[d]['gratings'] = dict(
                resp=f['tuning_mean'][:], sem=f['tuning_sem'][:], cond=f['orientations'][:].astype(float),
                rel=f['rel_r'][:], thresh=np.nanpercentile(f['rel_null'][:], NULL_PCTL, axis=1),
                osi=f['osi'][:], dsi=f['dsi'][:], pref=f['pref'][:])
        for m in MOVIES:
            R = movie_resp(d, m, OFFSET[(d, m)])
            func[d][m] = dict(resp=R.mean(axis=1), sem=R.std(axis=1) / np.sqrt(R.shape[1]),
                            cond=movie_src[d][m]['onsets'][0] - movie_src[d][m]['onsets'][0, 0],
                            rel=rel_movie[d][m], thresh=thresh_movie[d][m])
            del R
        with h5py.File(os.path.join(rec_dir(d, 'sparse_noise'), 'sparse_noise.h5'), 'r') as f:
            shape = tuple(f['shape'][()])
            S1, S2 = smooth_sta(f['STA1'][:], shape), smooth_sta(f['STA2'][:], shape)
            rel = rowcorr(S1, S2)
            null = np.concatenate([rowcorr(S1, S2[derangement(len(S1))]) for _ in range(N_SN_NULL)])
            del S1, S2
            func[d]['sparse_noise'] = dict(resp=smooth_sta(f['STA'][:], shape), rel=rel,
                                        thresh=np.full(len(rel), np.nanpercentile(null, NULL_PCTL)),
                                        lags=f['lags'][:], shape=shape)

    for m in MOVIES:
        n_f = min(func[d][m]['resp'].shape[1] for d in DAYS)
        for d in DAYS:
            for k in ('resp', 'sem', 'cond'):
                func[d][m][k] = func[d][m][k][..., :n_f]

    reliable = {d: {s: func[d][s]['rel'] > func[d][s]['thresh'] for s in STIMS} for d in DAYS}


    cells2 = s2p[DAYS[1]]['iscell_raw']
    cen2_all = w2_to_w1(s2p[DAYS[1]]['cen'][cells2])
    n2 = len(cells2)
    _, near = tree[DAYS[1]].query(c1, k=N_CANDIDATES + 1)
    cand = np.zeros((N, N_CANDIDATES), int)
    for k in range(N):
        others = [j for j in near[k] if j != idx[DAYS[1]][k]][:N_CANDIDATES - 1]
        cand[k] = [idx[DAYS[1]][k]] + others
    nearest_other = cand[:, 1]
    random_other = np.array([rng.choice(np.delete(np.arange(n2), idx[DAYS[1]][k])) for k in range(N)])


    def pair_r(s, rows1, rows2):
        return rowcorr(func[DAYS[0]][s]['resp'][rows1], func[DAYS[1]][s]['resp'][rows2])


    sim, sim_near, sim_rand, rank = {}, {}, {}, {}
    for s in STIMS:
        sim[s] = pair_r(s, idx[DAYS[0]], idx[DAYS[1]])
        sim_near[s] = pair_r(s, idx[DAYS[0]], nearest_other)
        sim_rand[s] = pair_r(s, idx[DAYS[0]], random_other)
        cand_r = np.stack([pair_r(s, idx[DAYS[0]], cand[:, j]) for j in range(N_CANDIDATES)], axis=1)
        rank[s] = 1 + np.sum(np.nan_to_num(cand_r[:, 1:], nan=-2) > np.nan_to_num(cand_r[:, [0]], nan=-2), axis=1).astype(float)
        rank[s][~np.isfinite(cand_r[:, 0])] = np.nan
    rel1 = {s: reliable[DAYS[0]][s][idx[DAYS[0]]] for s in STIMS}
    rel2 = {s: reliable[DAYS[1]][s][idx[DAYS[1]]] for s in STIMS}

    fig, axs = plt.subplots(1, len(STIMS), figsize=(11, 2.6), sharey=True)
    summary = []
    for ax, s in zip(axs, STIMS):
        p = rel1[s]
        ecdf(ax, sim[s][p], color=COLOR[s], label='matched partner')
        ecdf(ax, sim_near[s][p], color=INK2, ls='--', label='nearest other ROI')
        ecdf(ax, sim_rand[s][p], color=MUTED, ls=':', label='random cell')
        ax.set_xlim(-0.6, 1); ax.set_ylim(0, 1)
        ax.set_title('{} (n={})'.format(LABEL[s], p.sum()))
        ax.set_xlabel('cross-day r')
        summary.append({'stimulus': LABEL[s], 'n reliable on week 1': int(p.sum()),
                        'median r partner': np.nanmedian(sim[s][p]), 'median r nearest other': np.nanmedian(sim_near[s][p]),
                        'median r random': np.nanmedian(sim_rand[s][p]),
                        'partner beats nearest (frac)': np.nanmean((sim[s] > sim_near[s])[p]),
                        'partner ranks 1st of {} (frac)'.format(N_CANDIDATES): np.nanmean(rank[s][p] == 1),
                        'distance to nearest other (median px)': np.median(np.linalg.norm(cen2_all[nearest_other[p]] - c1[p], axis=1))})
    axs[0].set_ylabel('fraction of pairs')
    axs[0].legend(loc='upper left', fontsize=6.5)
    fig.suptitle('Week 1 cell vs week 2 partner, neighbor and random cell (week 1 cell reliable on week 1)', x=0.01, ha='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'diag9_partner_vs_neighbor')
    pd.DataFrame(summary).set_index('stimulus').round(3)


    fig, axs = plt.subplots(1, 3, figsize=(11, 2.8), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
    ax = axs[0]
    x = np.arange(len(STIMS))
    fr1 = [np.nanmean(rank[s][rel1[s]] == 1) for s in STIMS]
    ax.bar(x, fr1, 0.6, color=[COLOR[s] for s in STIMS])
    for xi, v in zip(x, fr1):
        ax.text(xi, v, '{:.0%}'.format(v), ha='center', va='bottom', fontsize=6.5, color=INK2)
    ax.axhline(1 / N_CANDIDATES, color=INK, lw=0.8)
    ax.text(x[-1] + 0.35, 1 / N_CANDIDATES, 'chance', va='bottom', ha='right', fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in STIMS], rotation=30, ha='right')
    ax.set_ylim(0, 1.08); ax.grid(axis='x', visible=False)
    ax.set_ylabel('fraction of week 1-reliable cells')
    ax.set_title('Partner is the best of {}'.format(N_CANDIDATES))

    q_rows = []
    for ax, (vals, xlabel) in zip(axs[1:], [(matched.p_overlap.to_numpy(), 'ROI overlap (quartile bins)'),
                                            (resid_d, 'centroid residual, px (quartile bins)')]):
        edges = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1])
        b = np.clip(np.digitize(vals, edges[1:-1]), 0, 3)
        centers = [np.median(vals[b == i]) for i in range(4)]
        for s in STIMS:
            p = rel1[s] & rel2[s]
            med = [np.nanmedian(sim[s][p & (b == i)]) if np.sum(p & (b == i)) >= 3 else np.nan for i in range(4)]
            ax.plot(centers, med, color=COLOR[s], marker='o', ms=4, lw=1.5, label=LABEL[s])
            q_rows.append({'stimulus': LABEL[s], 'metric': xlabel.split(' (')[0],
                        'Spearman rho': stats.spearmanr(vals[p], sim[s][p], nan_policy='omit')[0], 'n': int(p.sum())})
        ax.set_xlabel(xlabel); ax.set_ylim(0, 1)
        ax.set_ylabel('median cross-day r')
    axs[1].legend(loc='lower right', fontsize=6.5, ncol=2)
    axs[1].set_title('Similarity vs ROI overlap (reliable both days)')
    axs[2].set_title('Similarity vs residual (reliable both days)')
    fig.tight_layout()
    savefig(fig, 'diag10_rank_and_match_quality')
    pd.DataFrame(q_rows).pivot(index='stimulus', columns='metric', values='Spearman rho').round(3)


    S = np.stack([np.where(rel1[s], sim[s], np.nan) for s in STIMS], axis=1)
    mean_sim = np.nanmean(S, axis=1)
    has = np.isfinite(mean_sim)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axs[0]
    ax.imshow(norm01(img1) * 0.35, cmap='gray', vmin=0, vmax=1)
    ax.scatter(c1[~has, 0], c1[~has, 1], s=10, facecolor='none', edgecolor=MUTED, lw=0.5)
    sc = ax.scatter(c1[has, 0], c1[has, 1], c=mean_sim[has], cmap=SEQUENTIAL, vmin=0, vmax=1, s=14, edgecolor=SURFACE, lw=0.4)
    no_ticks(ax)
    fig.colorbar(sc, ax=ax, shrink=0.8, label='mean cross-day r')
    ax.set_title('Cross-day similarity across the field of view')
    ax = axs[1]
    gy = np.clip(((c1[:, 1] - off) / step).round().astype(int), 0, lc_reg.shape[0] - 1)
    gx = np.clip(((c1[:, 0] - off) / step).round().astype(int), 0, lc_reg.shape[1] - 1)
    lc_at = lc_reg[gy, gx]
    ax.scatter(lc_at[has], mean_sim[has], s=6, color=COLOR['gratings'], lw=0, alpha=0.7)
    ax.set_xlabel('local image r at the cell'); ax.set_ylabel('mean cross-day r')
    ok = has & np.isfinite(lc_at)
    ax.set_title('Function vs local image agreement (rho = {:.2f})'.format(stats.spearmanr(lc_at[ok], mean_sim[ok])[0]))
    fig.tight_layout()
    savefig(fig, 'diag11_similarity_map')


    u1, u2 = unmatched[DAYS[0]], unmatched[DAYS[1]]
    cu1, cu2 = s2p[DAYS[0]]['cen'][u1], w2_to_w1(s2p[DAYS[1]]['cen'][u2])
    d12, j12 = cKDTree(cu2).query(cu1)
    d21, j21 = cKDTree(cu1).query(cu2)
    mutual = (j21[j12] == np.arange(len(u1))) & (d12 < NEAR_MISS_PX)
    cand1, cand2 = u1[mutual], u2[j12[mutual]]
    ci1 = np.searchsorted(s2p[DAYS[0]]['iscell_raw'], cand1)
    ci2 = np.searchsorted(s2p[DAYS[1]]['iscell_raw'], cand2)
    assert np.array_equal(s2p[DAYS[0]]['iscell_raw'][ci1], cand1) and np.array_equal(s2p[DAYS[1]]['iscell_raw'][ci2], cand2)


    def mask_w1(d, r):
        st = s2p[d]['stat'][r]
        xy = np.stack([st['xpix'], st['ypix']], axis=1).astype(float)
        if d == DAYS[1]:
            xy = w2_to_w1(xy)
        return set(map(tuple, np.round(xy).astype(int)))


    def iou(r1, r2):
        a, b = mask_w1(DAYS[0], r1), mask_w1(DAYS[1], r2)
        return len(a & b) / len(a | b)


    iou_matched = np.array([iou(raw[DAYS[0]][k], raw[DAYS[1]][k]) for k in range(N)])
    iou_cand = np.array([iou(a, b) for a, b in zip(cand1, cand2)])
    cand_sim = {s: rowcorr(func[DAYS[0]][s]['resp'][ci1], func[DAYS[1]][s]['resp'][ci2]) for s in STIMS}
    cand_rel1 = {s: reliable[DAYS[0]][s][ci1] for s in STIMS}
    cand_rel2 = {s: reliable[DAYS[1]][s][ci2] for s in STIMS}

    fig, axs = plt.subplots(1, len(STIMS) + 1, figsize=(11, 2.5))
    ax = axs[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(iou_cand, bins=bins, color=AXIS, rwidth=0.9, label='candidates')
    ax.hist(iou_matched, bins=bins, histtype='step', color=COLOR['gratings'], lw=1.5, label='matched')
    ax.set_xlabel('mask IoU (week 1 frame)'); ax.set_ylabel('pairs')
    ax.legend(loc='upper right'); ax.set_title('Overlap')
    rows = []
    for ax, s in zip(axs[1:], STIMS):
        ecdf(ax, sim[s][rel1[s]], color=COLOR[s], label='matched')
        ecdf(ax, cand_sim[s][cand_rel1[s]], color=COLOR[s], ls='--', label='candidate')
        ecdf(ax, sim_near[s][rel1[s]], color=MUTED, ls=':', label='nearest other')
        ax.set_xlim(-0.6, 1); ax.set_ylim(0, 1)
        ax.set_title('{} (n={})'.format(LABEL[s], cand_rel1[s].sum()))
        ax.set_xlabel('cross-day r')
        hi = np.nanpercentile(sim_near[s][rel1[s]], 95)
        rows.append({'stimulus': LABEL[s], 'candidates reliable on week 1': int(cand_rel1[s].sum()),
                    'median r candidates': np.nanmedian(cand_sim[s][cand_rel1[s]]),
                    'median r matched': np.nanmedian(sim[s][rel1[s]]),
                    'median r nearest other': np.nanmedian(sim_near[s][rel1[s]]),
                    'candidates above 95th pct of nearest-other': np.nanmean(cand_sim[s][cand_rel1[s]] > hi),
                    'matched above 95th pct of nearest-other': np.nanmean(sim[s][rel1[s]] > hi)})
    axs[1].set_ylabel('fraction of pairs')
    axs[1].legend(loc='upper left', fontsize=6.5)
    fig.tight_layout()
    savefig(fig, 'diag12_missed_matches')

    candidates = pd.DataFrame({'day1_raw': cand1, 'day1_iscell': ci1, 'day2_raw': cand2, 'day2_iscell': ci2,
                            'distance_px': d12[mutual], 'mask_iou': iou_cand})
    for s in STIMS:
        candidates['xday_r_{}'.format(s)] = cand_sim[s]
        candidates['reliable_{}_wk1'.format(s)] = cand_rel1[s]
        candidates['reliable_{}_wk2'.format(s)] = cand_rel2[s]
    candidates.to_csv(os.path.join(OUT_DIR, 'candidate_missed_matches.csv'), index=False)
    print('{} candidate pairs (unmatched mutual nearest neighbors within {} px); {} pairs are matched now'.format(
        len(cand1), NEAR_MISS_PX, N))
    print('mask IoU: matched median {:.2f}, candidates median {:.2f}'.format(np.median(iou_matched), np.median(iou_cand)))
    pd.DataFrame(rows).set_index('stimulus').round(3)

    per_cell = pd.DataFrame({
        'unified_id': matched.unified_id, 'day1_iscell': idx[DAYS[0]], 'day2_iscell': idx[DAYS[1]],
        'x_week1': c1[:, 0], 'y_week1': c1[:, 1], 'p_overlap': matched.p_overlap, 'centroid_residual_px': resid_d,
        'area_ratio': area_ratio, 'patch_r': patch_r, 'shape_r': shape_r, 'local_image_r': lc_at,
    })
    for s in STIMS:
        per_cell['rate_{}_wk1'.format(s)] = rates[DAYS[0]][s][idx[DAYS[0]]]
        per_cell['rate_{}_wk2'.format(s)] = rates[DAYS[1]][s][idx[DAYS[1]]]
        per_cell['rel_{}_wk1'.format(s)] = func[DAYS[0]][s]['rel'][idx[DAYS[0]]]
        per_cell['rel_{}_wk2'.format(s)] = func[DAYS[1]][s]['rel'][idx[DAYS[1]]]
        per_cell['reliable_{}_wk1'.format(s)] = rel1[s]
        per_cell['reliable_{}_wk2'.format(s)] = rel2[s]
        per_cell['xday_r_{}'.format(s)] = sim[s]
        per_cell['xday_r_nearest_{}'.format(s)] = sim_near[s]
        per_cell['partner_rank_{}'.format(s)] = rank[s]
    per_cell['n_reliable_both'] = np.sum([rel1[s] & rel2[s] for s in STIMS], axis=0)
    per_cell['n_beaten_by_neighbor'] = np.sum([rel1[s] & (rank[s] > 1) for s in STIMS], axis=0)
    per_cell['n_reliable_wk1'] = np.sum([rel1[s] for s in STIMS], axis=0)
    per_cell.to_csv(os.path.join(OUT_DIR, 'alignment_per_cell.csv'), index=False)
    per_cell[['p_overlap', 'centroid_residual_px', 'area_ratio', 'patch_r', 'shape_r', 'n_reliable_both']].describe().round(2)


    def show_crop(ax, img, c, h=CROP):
        pts = crop_points(c, h)
        ax.imshow(norm01(sample(img, pts)), cmap='gray', vmin=0, vmax=1,
                extent=[c[0] - h - 0.5, c[0] + h + 0.5, c[1] + h + 0.5, c[1] - h - 0.5])
        no_ticks(ax)


    def plot_pair(r1, r2, header, ranks=None):
        d1, d2 = DAYS
        i1 = int(np.searchsorted(s2p[d1]['iscell_raw'], r1))
        i2 = int(np.searchsorted(s2p[d2]['iscell_raw'], r2))
        ca, cb = s2p[d1]['cen'][r1], s2p[d2]['cen'][r2]
        cb1 = w2_to_w1(cb)
        pair_sim = {s: rowcorr(func[d1][s]['resp'][[i1]], func[d2][s]['resp'][[i2]])[0] for s in STIMS}
        fig = plt.figure(figsize=(11, 4.9))
        gs = fig.add_gridspec(2, 24, hspace=0.6, wspace=1.4, left=0.02, right=0.99, top=0.86, bottom=0.1)

        ax = fig.add_subplot(gs[0, 0:4])
        show_crop(ax, img1, ca)
        ct = contours[d1][r1]; ax.plot(ct[:, 0], ct[:, 1], color=IMG_COLOR[d1], lw=1)
        ax.set_title(d1)
        ax = fig.add_subplot(gs[0, 4:8])
        show_crop(ax, img2, cb)
        ct = contours[d2][r2]; ax.plot(ct[:, 0], ct[:, 1], color=IMG_COLOR[d2], lw=1)
        ax.set_title('{} (own frame)'.format(d2))
        ax = fig.add_subplot(gs[0, 8:12])
        h = CROP
        ax.imshow(composite(patch_w1(img1, ca), patch_w2_in_w1(img2, ca)),
                extent=[ca[0] - h - 0.5, ca[0] + h + 0.5, ca[1] + h + 0.5, ca[1] - h - 0.5])
        a, b = contours[d1][r1], w2_to_w1(contours[d2][r2])
        ax.plot(a[:, 0], a[:, 1], color=IMG_COLOR[d1], lw=1)
        ax.plot(b[:, 0], b[:, 1], color=IMG_COLOR[d2], lw=1, ls='--')
        no_ticks(ax)
        ax.set_title('Overlay (wk 2 warped)')

        ax = fig.add_subplot(gs[0, 13:18])
        g = 'gratings'
        for d, i, color in [(d1, i1, DAY_COLOR[d1]), (d2, i2, COLOR[g])]:
            m, e = func[d][g]['resp'][i], func[d][g]['sem'][i]
            ax.plot(func[d][g]['cond'], m, color=color, marker='o', ms=3, lw=1.2, label=d)
            ax.fill_between(func[d][g]['cond'], m - e, m + e, color=color, alpha=0.2, lw=0)
        ax.set_xticks([0, 90, 180, 270]); ax.set_xlabel('direction (deg)'); ax.set_ylabel('spikes/s')
        ax.set_title('Gratings  r = {:.2f}'.format(pair_sim[g]))
        ax.legend(loc='upper right', fontsize=6)

        sn = 'sparse_noise'
        shape = func[d1][sn]['shape']
        sta1 = func[d1][sn]['resp'][i1].reshape(shape)
        pk = np.unravel_index(np.argmax(np.abs(sta1)), shape)
        for col, (d, i) in enumerate([(d1, i1), (d2, i2)]):
            ax = fig.add_subplot(gs[1, col * 4:col * 4 + 4])
            im = func[d][sn]['resp'][i].reshape(shape)
            clim = np.nanmax(np.abs(im)) or 1
            ax.imshow(im, cmap=DIVERGING, vmin=-clim, vmax=clim, interpolation='nearest')
            ax.plot(pk[1], pk[0], marker='+', ms=8, color=INK, mew=1)
            no_ticks(ax)
            ax.set_title('STA {}{}'.format(d, '  r = {:.2f}'.format(pair_sim[sn]) if col == 0 else ''))

        for j, m in enumerate(MOVIES):
            ax = fig.add_subplot(gs[1, 9 + 5 * j:13 + 5 * j])
            for d, i, color in [(d1, i1, DAY_COLOR[d1]), (d2, i2, COLOR[m])]:
                mm, e = func[d][m]['resp'][i], func[d][m]['sem'][i]
                t = func[d][m]['cond']
                ax.plot(t, mm, color=color, lw=0.9)
                ax.fill_between(t, mm - e, mm + e, color=color, alpha=0.2, lw=0)
            ax.set_xlabel('movie time (s)')
            if j == 0:
                ax.set_ylabel('spikes/s')
            ax.set_title('{}  r = {:.2f}'.format(LABEL[m], pair_sim[m]))
        return fig


    def plot_cell(k, tag=''):
        header = ['{}cell {}   ROI overlap {:.2f}'.format(tag + ': ' if tag else '', matched.unified_id[k], matched.p_overlap[k])]
        return plot_pair(raw[DAYS[0]][k], raw[DAYS[1]][k], header, ranks={s: rank[s][k] for s in STIMS})


    # order_strong = per_cell.assign(ms=np.nan_to_num(mean_sim)).sort_values(['n_reliable_both', 'ms'], ascending=False).index[:4]
    # sus = per_cell[(per_cell.n_reliable_wk1 >= 2)].assign(frac=lambda t: t.n_beaten_by_neighbor / t.n_reliable_wk1)
    # order_suspect = sus.sort_values(['frac', 'n_beaten_by_neighbor', 'p_overlap'], ascending=[False, False, True]).index[:3]
    # order_low = per_cell[per_cell.n_reliable_wk1 >= 2].sort_values('p_overlap').index[:3]
    # examples = [('strong', k) for k in order_strong] + [('suspect', k) for k in order_suspect] + [('low overlap', k) for k in order_low]
    # for tag, k in examples:
    #     fig = plot_cell(k, tag)
    #     savefig(fig, 'diag13_example_{}_{}'.format(tag.replace(' ', '_'), matched.unified_id[k]))
    #     plt.close(fig)

    Sc = np.stack([np.where(cand_rel1[s], cand_sim[s], np.nan) for s in STIMS], axis=1)
    n_ok = np.sum(np.isfinite(Sc), axis=1)
    score = np.where(n_ok >= 2, np.nanmean(Sc, axis=1), -np.inf)
    for c in np.argsort(-score)[:2]:
        fig = plot_pair(cand1[c], cand2[c], ['candidate (unmatched)', 'mask IoU {:.2f}'.format(iou_cand[c])])
        savefig(fig, 'diag13_example_candidate_{}_{}'.format(cand1[c], cand2[c]))
        plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('selected', type=str, default=None)
    parser.add_argument('compare_to_selected', type=str, default=None)
    args = parser.parse_args()

    if (args.selected is None) or (args.compare_to_selected is None):
        print('You need to select the directory of a recording where all stimuli were run in suite2p (i.e., a directory with all tifs).')
        selected = select_directory('Select first recording directory.')
        compare_to_selected = select_directory('Choose a directory to compare to the selected directory.')
    else:
        selected = args.selected
        compare_to_selected = args.compare_to_selected

    main(selected, compare_to_selected)