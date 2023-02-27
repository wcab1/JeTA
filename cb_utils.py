"""
Original script written by Claire Bedbrook .
"""

import matplotlib.pyplot as plt
import skimage.io
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter # smooth lines

from matplotlib import cm
import pandas as pd
import jb_utils as jb
import cb_utils as cb

import scipy.fftpack # fourier transform
import csv
import pandas as pd
from tqdm import tqdm
###############################################################################
###############################################################################


    
def fourier_transform_half_1(signal, jtype, n, frame_start, frame_end, f_ratio=1):
    
    plt.figure('Fourier Transform')

    ft = np.fft.fft(signal)
    mag_spectrum = np.abs(ft)
    sm_mag_spectrum = savgol_filter(mag_spectrum, 17,3)
    freq = np.linspace(0, n, len(mag_spectrum))
    num_freq_bins = int(len(freq)* f_ratio)

    plt.plot(freq[:num_freq_bins], sm_mag_spectrum[:num_freq_bins],'.-')
    
    plt.xlabel("Frequency(Hz)")
    plt.title(jtype)
    
    plt.xlim([0, 5])
    plt.ylim([0,55])
    plt.minorticks_on()
    
    plt.grid(b=True, which='major')
    plt.show()
    
    jellyfish1 = 'Medusa'
    jellyfish2 = 'Ephyra'
    path_save2 = 'C:/Users/wcab2/Documents/Jellyfish-master/CSV/FT/'
    df = pd.DataFrame(sm_mag_spectrum[:num_freq_bins])
    
    for item in range(frame_start, frame_end):
        
            df.to_csv(path_save2 +'Half1' + str(jellyfish1) + '_1' + '.csv', index=False)
        
def fourier_transform_half_2(signal, jtype, n, frame_start, frame_end, f_ratio=1):
    plt.figure('Fourier Transform')

    ft = np.fft.fft(signal)
    mag_spectrum = np.abs(ft)
    sm_mag_spectrum = savgol_filter(mag_spectrum, 17,3)
    freq = np.linspace(0, n, len(mag_spectrum))
    num_freq_bins = int(len(freq)* f_ratio)

    plt.plot(freq[:num_freq_bins], sm_mag_spectrum[:num_freq_bins],'.-')
    
    plt.xlabel("Frequency(Hz)")
    plt.title(jtype)
    
    plt.xlim([0, 5])
    plt.ylim([0,55])
    plt.minorticks_on()
    
    plt.grid(b=True, which='major')
    plt.show()
    
    jellyfish1 = 'Medusa'
    jellyfish2 = 'Ephyra'
    path_save2 = 'C:/Users/wcab2/Documents/Jellyfish-master/CSV/FT/'
    df = pd.DataFrame(sm_mag_spectrum[:num_freq_bins])
    
    for item in range(frame_start, frame_end):
        
            df.to_csv(path_save2 +'Half2' + str(jellyfish1) + '_1' + '.csv', index=False)    
    
def peak_counter(df, jelly, thresh, fps, lam_big, lam_small, data_start, data_end, n):
    """
    Input:
        df: dataframe with each column for one jellyfish and each row for a 
                single frame's measured pixel intentisty 
        jelly: jellyfish number
        thresh: threshold for normalized activtiy trace to i.d. peaks
        fps: frames per second
        lam_big: filtering parameter for basline
        lam_small: filtering parameter for trace smoothing
        data_start: first frame (time = 0)
        data_end: last frame of jump (time = 20 min)
        n: jump number
    
    Returns:
        peak_dist_lst: distance (# frames) between peaks
        peak_dist_t_lst: distance (time) between peaks, i.e. IPI
        peak_numb_lst: # number of peaks within time period (20 min)
        peak_inds: array of frame number with pulse peak
    """
    
    # Arrays for relevant peak counting values
    peak_dist_lst = [] # distance (# frames) between peaks
    peak_dist_t_lst = [] # distance (time) between peaks, i.e. IPI
    peak_numb_lst = [] # number of peaks
    
    # check if the file has data for analysis                  
    if np.isnan(df[jelly][1]) == True:
        peak_dist_lst.append(0)
        peak_numb_lst.append(0)
        print('no data, cannot count peaks')
    
    # if data, then continue   
    else:                
        # Scale by activity trace by 1000 for filtering purposes
        trace = 1000 * df[jelly][data_start:data_end].values
        
        # Array of frames in the file
        frames = np.array(range(len(df[jelly][data_start:data_end])))
        # frames= np.array(range(len(df[jelly][0:790])))
        
        # Convert frames to time in seconds 
        time = frames * 1.0 / (fps) # time in s
        
        # Find the baseline by smoothing the trace  
        baseline = jb.nw_kernel_smooth(frames, frames, trace, jb.epan_kernel, lam_big)
        
        # Find the max based on the delta from baseline to max hight of the pulse
        # from a section of the trace
        delta_max1 = np.max(trace[0:1000] - baseline[0:1000])
        
        delta_max = delta_max1      
        # Use the delta between baseline and top of the pulse to normalize the 
        # trace:
        trace_norm = ((trace - baseline) / ((baseline + delta_max) - baseline))
        
        # line_smooth = make_interp_spline(time, trace_norm, k=3)
        trace_norm_smooth = savgol_filter(trace_norm, 9, 5)
        
        # Find max and min of the trace to find peaks 
        # max_inds = scipy.signal.argrelmax(trace_norm, order=5) 
        max_inds = scipy.signal.argrelmax(trace_norm_smooth, order=5) 

        ####### Replaced trace_norm with trace_norm_smooth for all instances####
        # Find places where two contiguous points are equal- i.e. looking
        # for the flat peaks
        plataeu_peak= []
        for i in range(len(trace)-1):
            if (trace[i+1] - trace[i] == 0.0) & (trace_norm_smooth[i] > thresh):
                plataeu_peak.append(i)
                max_inds = np.column_stack((max_inds, i))
        
        # sort full list of max_inds with appended plataeu_peak 's
        max_inds = np.sort(max_inds[0])
                                
        # Find peak_inds: go through list of all max_inds and exclude max 
        # within four frames of each other
        peak_inds = []
        for i in range(len(max_inds) - 1):
            if (trace_norm_smooth[max_inds][i] > thresh) & \
                    (max_inds[i+1] != max_inds[i]) & \
                    (max_inds[i+1] != max_inds[i] + 1) & \
                    (max_inds[i+1] != max_inds[i] + 2) & \
                    (max_inds[i+1] != max_inds[i] + 3) & \
                    (max_inds[i+1] != max_inds[i] + 4):
                peak_inds.append(max_inds[i])
        
        # Plot the normalized trace with max min
        plt.figure('Hour_' + str(n))
        plt.plot(time,trace_norm_smooth , "ob-", markersize = 3, alpha=1,)
        
        ######################################################################
        
        exp = np.array([time, trace_norm_smooth])
        exp_t = exp.T
        path_save2 = 'C:/Users/wcab2/Documents/Jellyfish-master/CSV/Intensity Graphs/'
        df = pd.DataFrame(np.array(exp_t))
        
    
        # for item in range(data_start, data_end):
        #     df.to_csv(path_save2 +'LIVE' + '_3' + 'blank'+'.csv', index=False)
        
        
        ######################################################################
        ax = plt.gca()
        # ax.set_xlim([xmin, xmax])
        ax.set_ylim([-3, 3])
       
        
        plt.plot(time[peak_inds], trace_norm_smooth[peak_inds], '.r')
        
        plt.minorticks_on()
        plt.grid(b=True, which='major')
        
        plt.xlabel('Time [s]', fontsize=25)
        plt.ylabel('Normalized Intensity', fontsize=25)
        
        ################################################################
        
        # fourier_transform_half_1(trace_norm_smooth, "Jellyfish Fourier Transform LIVE3 Half 1", fps, 0, 1824, 0.5)
    
        ################################################################
        
        # Compute the total number of peaks
        peak_numb_lst.append(len(peak_inds))  
        
        # Compute distance between peaks (i.e. IPI)
        for i in range(len(peak_inds)):
             
            if i < len(peak_inds)-1:
                peak_dists = peak_inds[i + 1] -  peak_inds[i]
                peak_dists_t = time[peak_inds][i]
                peak_dist_lst.append(peak_dists / (15.0))  # for 15 fps
                peak_dist_t_lst.append(peak_dists_t)
    plt.show()
    
    path_save2 = 'C:/Users/wcab2/Documents/Jellyfish-master/example_data/TXT/intensity_data/'
    df = pd.DataFrame(trace_norm_smooth)
    
    for item in range(data_start, data_end):
        
            df.to_csv(path_save2 +'Half1' + '.csv', index=False)
    
    return [peak_dist_lst, peak_dist_t_lst, peak_numb_lst, peak_inds] 

def peak_counter_2(df, jelly, thresh, fps, lam_big, lam_small, data_start, data_end, n):
    """
    Input:
        df: dataframe with each column for one jellyfish and each row for a 
                single frame's measured pixel intentisty 
        jelly: jellyfish number
        thresh: threshold for normalized activtiy trace to i.d. peaks
        fps: frames per second
        lam_big: filtering parameter for basline
        lam_small: filtering parameter for trace smoothing
        data_start: first frame (time = 0)
        data_end: last frame of jump (time = 20 min)
        n: jump number
    
    Returns:
        peak_dist_lst: distance (# frames) between peaks
        peak_dist_t_lst: distance (time) between peaks, i.e. IPI
        peak_numb_lst: # number of peaks within time period (20 min)
        peak_inds: array of frame number with pulse peak
    """
    
    # Arrays for relevant peak counting values
    peak_dist_lst = [] # distance (# frames) between peaks
    peak_dist_t_lst = [] # distance (time) between peaks, i.e. IPI
    peak_numb_lst = [] # number of peaks
    
    # check if the file has data for analysis                  
    if np.isnan(df[jelly][1]) == True:
        peak_dist_lst.append(0)
        peak_numb_lst.append(0)
        print('no data, cannot count peaks')
    
    # if data, then continue   
    else:                
        # Scale by activity trace by 1000 for filtering purposes
        trace = 1000 * df[jelly][data_start:data_end].values
        
        # Array of frames in the file
        frames = np.array(range(len(df[jelly][data_start:data_end])))
        # frames= np.array(range(len(df[jelly][0:790])))
        
        # Convert frames to time in seconds 
        time = frames * 1.0 / (fps) # time in s
        
        # Find the baseline by smoothing the trace  
        baseline = jb.nw_kernel_smooth(frames, frames, trace, jb.epan_kernel, lam_big)
        
        # Find the max based on the delta from baseline to max hight of the pulse
        # from a section of the trace
        delta_max1 = np.max(trace[0:1000] - baseline[0:1000])
        
        delta_max = delta_max1      
        # Use the delta between baseline and top of the pulse to normalize the 
        # trace:
        trace_norm = ((trace - baseline) / ((baseline + delta_max) - baseline))
        
        # line_smooth = make_interp_spline(time, trace_norm, k=3)
        trace_norm_smooth2 = savgol_filter(trace_norm, 11, 5)
        
        # Find max and min of the trace to find peaks 
        # max_inds = scipy.signal.argrelmax(trace_norm, order=5) 
        max_inds = scipy.signal.argrelmax(trace_norm_smooth2, order=5) 

        ####### Replaced trace_norm with trace_norm_smooth for all instances####
        # Find places where two contiguous points are equal- i.e. looking
        # for the flat peaks
        plataeu_peak= []
        for i in range(len(trace)-1):
            if (trace[i+1] - trace[i] == 0.0) & (trace_norm_smooth2[i] > thresh):
                plataeu_peak.append(i)
                max_inds = np.column_stack((max_inds, i))
        
        # sort full list of max_inds with appended plataeu_peak 's
        max_inds = np.sort(max_inds[0])
                                
        # Find peak_inds: go through list of all max_inds and exclude max 
        # within four frames of each other
        peak_inds = []
        for i in range(len(max_inds) - 1):
            if (trace_norm_smooth2[max_inds][i] > thresh) & \
                    (max_inds[i+1] != max_inds[i]) & \
                    (max_inds[i+1] != max_inds[i] + 1) & \
                    (max_inds[i+1] != max_inds[i] + 2) & \
                    (max_inds[i+1] != max_inds[i] + 3) & \
                    (max_inds[i+1] != max_inds[i] + 4):
                peak_inds.append(max_inds[i])
        
        # Plot the normalized trace with max min
        plt.figure('Half 2')
        plt.plot(time, trace_norm_smooth2 , "ob-", markersize = 3, alpha=1,)
        
        ######################################################################
    
        ax = plt.gca()
        # ax.set_xlim([xmin, xmax])
        ax.set_ylim([-1, 1])
        
        plt.plot(time[peak_inds], trace_norm_smooth2[peak_inds], '.r')
        
        plt.minorticks_on()
        plt.grid(b=True, which='major')
        
        plt.xlabel('Time [s]', fontsize=25)
        plt.ylabel('Normalized Intensity', fontsize=25)

        
        # fourier_transform_half_2(trace_norm_smooth2, "Jellyfish Fourier Transform LIVE1 Half 2", fps, 1795, 3589, 0.5)

        # Compute the total number of peaks
        peak_numb_lst.append(len(peak_inds))  
        
        # Compute distance between peaks (i.e. IPI)
        for i in range(len(peak_inds)):
             
            if i < len(peak_inds)-1:
                peak_dists = peak_inds[i + 1] -  peak_inds[i]
                peak_dists_t = time[peak_inds][i]
                peak_dist_lst.append(peak_dists / (15.0))  # for 15 fps
                peak_dist_t_lst.append(peak_dists_t)
    plt.show()
    return [peak_dist_lst, peak_dist_t_lst, peak_numb_lst, peak_inds] 

def df_maker_IPI_time_stamp(p_lst, n):
    """
    make dataframe for single jellyfish of inter-peak distance with time-stamp
    (i.e. IPI times)
    Input:
        p_lst: array of frame number with pulse peak for each jump (hour)
        n: item, jump (hour) 
    Output:
        df: dataframe for each jellyfish for each hour with time of each pulse
            and inter-pulse interval (i.e. IPI times)
    """
    columns = len(p_lst)
    i_test = []
    
    for i in range(columns):
        i_test.append(len(p_lst[i]))
    
    # Find the jump with the most IPI (rows) this will be the length of the df
    ind = np.max(i_test)
    
    # Add NaN to each column until length of all columns are equale
    for col in range(columns):
        if len(p_lst[col]) < ind:
            for w in range(ind - len(p_lst[col])):
                p_lst[col].append('NaN')
                
    df = pd.DataFrame(index=range(ind))
    
    n = n + 1
    df['time_J' + str(n-1)] = p_lst[2 * n - 2]
    df['IPI_J' + str(n-1)] = p_lst[2 * n-1]   
    return df

def df_maker_peak_time(p_lst, n):
    """
    make dataframe for each jellyfish/each hour with the frame number of each pulse
    
    Input:
        p_lst: array of frame number with pulse peak for each jump (hour)
        n: item, jump (hour) 
    Output:
        df: dataframe for single jellyfish of for single jellyfish of IPI times
            each column is jump (hour)
    """
    # Prep df for peak time 
    columns = len(p_lst)
    i_test = []
    for i in range(columns):
        i_test.append(len(p_lst[i]))
    
    ind = np.max(i_test)
    
    for col in range(columns):
        if len(p_lst[col]) < ind:
            for w in range(ind - len(p_lst[col])):
                p_lst[col].append('NaN')
                
    df = pd.DataFrame(index=range(ind))
       
    df['Peak_frame_J' + str(n)] = p_lst[n]
        
    return df  