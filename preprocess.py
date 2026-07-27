

import os
import numpy as np
import tifffile
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats


import OMSI
import oasis


def load_twop(suite2p_dir, neu_correction=0.7, use_omsi=True):

    F_path = os.path.join(suite2p_dir, 'F.npy')
    Fneu_path = os.path.join(suite2p_dir, 'Fneu.npy')
    iscell_path = os.path.join(suite2p_dir, 'iscell.npy')
    spikes_path = os.path.join(suite2p_dir, 'spks.npy')

    F = np.load(F_path, allow_pickle=True)
    Fneu = np.load(Fneu_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)
    spks = np.load(spikes_path, allow_pickle=True)


    usecells = iscell[:, 0] == 1

    F = F[usecells, :]
    Fneu = Fneu[usecells, :]
    s2p_spks = spks[usecells, :]
    nCells = np.size(F, 0)
    lenT = np.size(F, 1)

    norm_F = np.zeros([nCells, lenT])
    raw_dFF = np.zeros([nCells, lenT])
    norm_dFF = np.zeros([nCells, lenT])
    norm_F0 = np.zeros(nCells)
    raw_F0 = np.zeros(nCells)
    denoised_dFF = np.zeros([nCells, lenT])

    for c in range(nCells):

        F_cell = F[c, :].copy()
        F_cell_neu = Fneu[c, :].copy()

        _lo, _hi = np.nanpercentile(F_cell, [1, 99])
        _counts, _edges = np.histogram(F_cell, bins=300, range=(_lo, _hi))
        _pk = np.argmax(_counts)
        _f0_raw = 0.5 * (_edges[_pk] + _edges[_pk + 1])

        _raw_dFF = (F_cell - _f0_raw) / _f0_raw

        _normF = F_cell - neu_correction * F_cell_neu + neu_correction * np.nanmean(F_cell_neu)

        _lo_n, _hi_n = np.nanpercentile(_normF, [1, 99])
        _counts_n, _edges_n = np.histogram(_normF, bins=300, range=(_lo_n, _hi_n))
        _pk_n = np.argmax(_counts_n)
        _f0_norm = 0.5 * (_edges_n[_pk_n] + _edges_n[_pk_n + 1])

        norm_dFF[c, :] = (_normF - _f0_norm) / _f0_norm

    if use_omsi:
        omsi_results = OMSI.deconv(F=F, Fneu=Fneu, hz=30.0)
        spike_times = omsi_results['spike_times']
        spike_train = omsi_results['spike_train']

    elif not use_omsi:
        spike_train = np.zeros([nCells, lenT])
        for c in range(nCells):
            g = oasis.functions.estimate_time_constant(norm_dFF[c, :].copy(), 1)
            denoised_dFF[c, :], spike_train[c, :] = oasis.oasisAR1(norm_dFF[c, :].copy(), g)
            spike_times = None

    twop_dict = {
        'F': F,
        'Fneu': Fneu,
        'iscell': iscell,
        's2p_spks': s2p_spks,
        'norm_dFF': norm_dFF,
        'raw_dFF': raw_dFF,
        'denoised_dFF': denoised_dFF,
        'spike_times': spike_times,
        'spike_train': spike_train
    }

    return twop_dict


def load_stimulus_info(stim_type=1):
    # stim type should be 1, 2, or 3 for natural movie number

    if stim_type == 1:
        stim_info = {
            'stim_name': 'natural_movie_one',
            'stim_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_stimuli/natural_movie_one.tif',
            'stim_n_frames': 900,
            'stim_fps': 30.0
        }
    elif stim_type == 2:
        stim_info = {
            'stim_name': 'natural_movie_two',
            'stim_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_stimuli/natural_movie_two.tif',
            'stim_n_frames': 900,
            'stim_fps': 30.0
        }
    elif stim_type == 3:
        stim_info = {
            'stim_name': 'natural_movie_three',
            'stim_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_stimuli/natural_movie_three.tif',
            'stim_n_frames': 3600,
            'stim_fps': 30.0
        }


    stim_stack = tifffile.imread(stim_info['stim_path'])

    stimT = np.arange(0, stim_info['stim_n_frames']) / stim_info['stim_fps']

    stim_info['stim_stack'] = stim_stack
    stim_info['stimT'] = stimT

    return stim_info


def open_dlc_h5(dlc_path, h5key=None):
    if h5key is None:
        pts = pd.read_hdf(dlc_path)
    else:
        pts = pd.read_hdf(dlc_path, key=h5key)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names


def fit_ellipse(x, y):

        meanX = np.mean(x)
        meanY = np.mean(y)
        x = x - meanX
        y = y - meanY

        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        a = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]

        Q = np.array([[a, b/2],[b/2, c]])
        eig_val, eig_vec = np.linalg.eig(Q)

        if eig_val[0] < eig_val[1]:
            angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        else:
            angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])

        angle_from_x = angle_to_x
        orientation_rad = 0.5 * np.arctan2(b, (c-a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

        a, b, c, d, e = [
            a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
            0,
            a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
            d*cos_phi - e*sin_phi,
            d*sin_phi + e*cos_phi
            ]

        meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                        sin_phi*meanX + cos_phi*meanY]

        test = a*c

        if test > 0:

            if a<0:
                a, c, d, e = [-a, -c, -d, -e]

            X0 = meanX - d/2/a
            Y0 = meanY - e/2/c
            F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
            a = np.sqrt(F/a)
            b = np.sqrt(F/c)
            long_axis = 2*np.maximum(a,b)
            short_axis = 2*np.minimum(a,b)

            R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
            P_in = R @ np.array([[X0],[Y0]])
            X0_in = P_in[0][0]
            Y0_in = P_in[1][0]

            ellipse_dict = {
                'X0':X0,
                'Y0':Y0,
                'F':F,
                'a':a,
                'b':b,
                'long_axis':long_axis/2,
                'short_axis':short_axis/2,
                'angle_to_x':angle_to_x,
                'angle_from_x':angle_from_x,
                'cos_phi':cos_phi,
                'sin_phi':sin_phi,
                'X0_in':X0_in,
                'Y0_in':Y0_in,
                'phi':orientation_rad
            }

        else:

            dict_keys = ['X0','Y0','F','a','b','long_axis',
                         'short_axis','angle_to_x','angle_from_x',
                         'cos_phi','sin_phi','X0_in','Y0_in','phi']
            dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

            ellipse_dict = dict(zip(dict_keys, dict_vals))
        
        return ellipse_dict


def apply_liklihood_thresh(x, l, threshold=0.99):

    thresh_arr = (l>threshold).astype(float).values
    x_vals1 = x.copy().values

    x_vals2 = pd.DataFrame((x_vals1 * thresh_arr), columns=x.columns)
    x_vals2[x_vals2==0.] = np.nan

    x_vals = x_vals2.copy()

    return x_vals


def split_xyl(xyl):
    names = list(xyl.columns.values)

    x_locs = []
    y_locs = []
    l_locs = []

    for loc_num in range(0, len(names)):
        loc = names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            l_locs.append(loc)

    x_vals = xyl[x_locs]
    y_vals = xyl[y_locs]
    l_vals = xyl[l_locs]

    return x_vals, y_vals, l_vals


def track_pupil(rec_dir, eye_dlc_h5):

    eye_dist_thresh = 20
    eye_pxl2cm = 24
    likelihood_thresh = 0.6
    eye_trackable_N = 6
    eye_calibration_N = 8
    eye_ellipse_thresh = 0.85

    pdf_name = 'eye_tracking_figs.pdf'
    pdf = PdfPages(os.path.join(rec_dir, pdf_name))

    xyl, _ = open_dlc_h5(eye_dlc_h5)
    x_vals, y_vals, likelihood = split_xyl(xyl)

    x_vals = apply_liklihood_thresh(
        x_vals, likelihood, threshold=likelihood_thresh
    )
    y_vals = apply_liklihood_thresh(
        y_vals, likelihood, threshold=likelihood_thresh
    )

    pupil_count = np.sum(likelihood >= likelihood_thresh, 1)
    usegood_eye = pupil_count >= eye_trackable_N
    usegood_eyecalib = pupil_count >= eye_calibration_N

    print(' !!  N={}/{} frames dropped for not meeting required number of tracked points ({}).'.format(
        np.sum(~usegood_eye), len(pupil_count), eye_trackable_N
    ))

    std_thresh_x = np.empty(np.shape(x_vals))
    std_thresh_y = np.empty(np.shape(y_vals))

    for point_loc in range(0,np.size(x_vals, 1)):
        _val = x_vals.iloc[:,point_loc]
        std_thresh_x[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / eye_pxl2cm) > eye_dist_thresh

    for point_loc in range(0,np.size(x_vals, 1)):
        _val = y_vals.iloc[:,point_loc]
        std_thresh_y[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / eye_pxl2cm) > eye_dist_thresh

    std_thresh_x = np.nanmean(std_thresh_x, 1)
    std_thresh_y = np.nanmean(std_thresh_y, 1)
    x_vals[std_thresh_x > 0] = np.nan
    y_vals[std_thresh_y > 0] = np.nan

    num_removed_for_dist_std = np.sum((std_thresh_x > 0) * (std_thresh_y > 0))
    print(' !!  N={}/{} frames dropped for distance std threshold.'.format(num_removed_for_dist_std, np.size(x_vals, 0)))

    ellipse = np.empty([len(usegood_eye), 14])

    cols = [
        'X0',              
        'Y0',              
        'F',               
        'a',               
        'b',               
        'long_axis',       
        'short_axis',      
        'angle_to_x',      
        'angle_from_x',    
        'cos_phi',         
        'sin_phi',          
        'X0_in',            
        'Y0_in',            
        'phi'               
    ]

    linalgerror = 0
    for step in tqdm(range(0,len(usegood_eye))):
        
        if usegood_eye[step] == True:
            
            try:

                e_t = fit_ellipse(x_vals.iloc[step].values,
                                        y_vals.iloc[step].values)
                
                ellipse[step] = [
                    e_t['X0'],                 
                    e_t['Y0'],                 
                    e_t['F'],                  
                    e_t['a'],                  
                    e_t['b'],                  
                    e_t['long_axis'],          
                    e_t['short_axis'],         
                    e_t['angle_to_x'],         
                    e_t['angle_from_x'],       
                    e_t['cos_phi'],            
                    e_t['sin_phi'],             
                    e_t['X0_in'],               
                    e_t['Y0_in'],               
                    e_t['phi']                  
                ]
            
            except np.linalg.LinAlgError as e:

                linalgerror = linalgerror + 1
                ellipse[step] = list(np.ones([len(cols)]) * np.nan)
        
        elif usegood_eye[step] == False:

            ellipse[step] = list(np.ones([len(cols)]) * np.nan)

    print('LinAlg error count = ' + str(linalgerror))

    ellipticity_test = ((ellipse[:,6] / ellipse[:,5]) < eye_ellipse_thresh)
    usegood_ellipcalb = np.where((usegood_eyecalib == True) & ellipticity_test)

    num_removed_for_ellipticity = np.sum(~ellipticity_test)
    print(' !!  N={}/{} frames dropped because of ellipticity threshold.'.format(num_removed_for_ellipticity, np.sum(usegood_eyecalib)))
    
    f_lim = 50000
    if np.size(usegood_ellipcalb,1) > f_lim:
        shortlist = sorted(np.random.choice(usegood_ellipcalb[0],
                            size=f_lim, replace=False))
    else:
        shortlist = usegood_ellipcalb
    
    A = np.vstack([np.cos(ellipse[shortlist,7]),
                    np.sin(ellipse[shortlist,7])])
    b = np.expand_dims(np.diag(A.T @ np.squeeze(ellipse[shortlist, 11:13].T)), axis=1)
    cam_cent = np.linalg.inv(A @ A.T) @ A @ b
    
    ellipticity = (ellipse[shortlist,6] / ellipse[shortlist,5]).T
    
    try:
        scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *\
        (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=0)))\
        / np.sum(1 - (ellipticity)**2)
    
    except ValueError:
        scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *\
        (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=1)))\
        / np.sum(1 - (ellipticity)**2)
    
                                    
    theta = np.arcsin((ellipse[:,11] - cam_cent[0]) / scale)

                                
    phi = np.arcsin((ellipse[:,12] - cam_cent[1]) / np.cos(theta) / scale)


    ellipse_dict = {
        'theta':list(theta),
        'phi':list(phi),
        'longaxis':list(ellipse[:,5]),
        'shortaxis':list(ellipse[:,6]),
        'X0':list(ellipse[:,11]),
        'Y0':list(ellipse[:,12]),
        'ellipse_phi':list(ellipse[:,7]),
                       
        'cam_center_x': cam_cent[0,0],
        'cam_center_y': cam_cent[1,0]
    }
    
    fig1, [[ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]] = plt.subplots(2,4, figsize=(15,5.5), dpi=300)

                             
    ax1.plot(pupil_count[0:-1:10])
    ax1.set_title('{:.3}% good'.format(np.mean(usegood_eye)*100))
    ax1.set_ylabel('num good pupil points')
    ax1.set_xlabel('every 10th frame')

                                  
    ax2.hist(pupil_count, bins=9, range=(0,9), density=True)
    ax2.set_xlabel('num good eye points')
    ax2.set_ylabel('fraction of frames')

                                     
    ax3.plot(np.rad2deg(theta)[0:-1:10])
    ax3.set_title('theta')
    ax3.set_ylabel('deg')
    ax3.set_xlabel('every 10th frame')

                                   
    ax4.plot(np.rad2deg(phi)[0:-1:10])
    ax4.set_title('phi')
    ax4.set_ylabel('deg')
    ax4.set_xlabel('every 10th frame')

    fig_dwnsmpl = 100

    try:
                             
        ax5.hist(ellipticity, density=True)
        ax5.set_title('ellipticity; thresh='+str(eye_ellipse_thresh))
        ax5.set_ylabel('ellipticity')
        ax5.set_xlabel('fraction of frames')
        
                                     
        w = ellipse[:,7]

        artc_x = []
        artc_y = []

        for i in range(0,len(usegood_ellipcalb)):

            _show = usegood_ellipcalb[i::fig_dwnsmpl]

            ax6.plot((ellipse[_show,11] + [-5 * np.cos(w[_show]),\
                        5 * np.cos(w[_show])]),\
                        (ellipse[_show,12] + [-5*np.sin(w[_show]),\
                        5*np.sin(w[_show])]))
            artc_x.append((ellipse[_show,11] + [-5 * np.cos(w[_show]),5*np.cos(w[_show])]))
            artc_y.append((ellipse[_show,12] + [-5*np.sin(w[_show]), 5*np.sin(w[_show])]))

        ax6.plot(cam_cent[0], cam_cent[1], 'r*')
        ax6.set_title('eye axes relative to center')


    except Exception as e:
        print('Figure error in plots of ellipticity and axes relative to center')
        print(e)
        
    try:
        
        xvals = np.linalg.norm(ellipse[usegood_eyecalib, 11:13].T - cam_cent, axis=0)

        yvals = scale * np.sqrt( 1 - (ellipse[usegood_eyecalib, 6]\
                                    / ellipse[usegood_eyecalib, 5]) **2)

        calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)

        slope, _, r_value, _, _ = scipy.stats.linregress(xvals[calib_mask],
                                                            yvals[calib_mask].T)
    
    except ValueError:
        print('No good frames that meet criteria... check DLC tracking!')

    ellipse_dict['scale'] = float(scale)
    ellipse_dict['regression_r'] = float(r_value)
    ellipse_dict['regression_m'] = float(slope)

    try:
        ax7.plot(xvals[::fig_dwnsmpl],
                    yvals[::fig_dwnsmpl], '.', markersize=1)
        ax7.plot(np.linspace(0,50), np.linspace(0,50), 'r')
        ax7.set_title('scale={:.3} r={:.3} m={:.3}'.format(scale, r_value, slope))
        ax7.set_xlabel('pupil camera dist')
        ax7.set_ylabel('scale * ellipticity')

                                      
        delta = (cam_cent - ellipse[:, 11:13].T)

        _useec = usegood_eyecalib[::fig_dwnsmpl]
        _use3 = np.squeeze(usegood_ellipcalb)[::fig_dwnsmpl]

        ax8.plot(np.linalg.norm(delta[:,_useec], 2, axis=0),\
                ((delta[0, _useec].T * np.cos(ellipse[_useec, 7]))\
                + (delta[1, _useec].T * np.sin(ellipse[_useec, 7])))\
                / np.linalg.norm(delta[:, _useec], 2, axis=0).T,\
                'y.', markersize=1)

        ax8.plot(np.linalg.norm(delta[:,_use3], 2, axis=0),\
                ((delta[0, _use3].T * np.cos(ellipse[_use3,7]))\
                + (delta[1, _use3].T * np.sin(ellipse[_use3, 7])))\
                / np.linalg.norm(delta[:, _use3], 2, axis=0).T,\
                'r.', markersize=1)

        ax8.set_title('camera center calibration')
        ax8.set_ylabel('abs([PC-EC]).[cos(w);sin(w)]')
        ax8.set_xlabel('abs(PC-EC)')

        patch0 = mpatches.Patch(color='y', label='all pts')
        patch1 = mpatches.Patch(color='y', label='calibration pts')
        plt.legend(handles=[patch0, patch1])

    except Exception as e:
        print(e)
        print('Error in scale, center, and calibration figures. Skipping these for now')
    
    fig1.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close()

    for k,v in ellipse_dict.items():
        ellipse_dict[k] = np.array(v)

    ellipse_dict['axes_rel_cent_x'] = artc_x
    ellipse_dict['axes_rel_cent_y'] = artc_y
    ellipse_dict['camcent'] = cam_cent

    return xyl, ellipse_dict


def main():

    


if __name__ == '__main__':
    main()