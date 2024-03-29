"""
Original Script written by Claire Bebrook.
"""

import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
# load custom utils
import jb_utils as jb
import cb_utils as cb

           
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Close all figures
plt.close('all')

# define path with folder of images for analysis
path = 'C:/Users/wcab2/Documents/Jellyfish-master/example_data/TXT/'

###############################################################################
###############################################################################
# define date, camera, and state (night/day) for analysis
camera = 'cam1'
state = 'Smell_Main'
date = '20170611'

###############################################################################
###############################################################################
# define path for initial frame for each hour of recoding for analysis (12 hours)
jfile = [path + str(date) + '/' + str(camera) + '/' + 'Smell_Main' + '/' + str(date) + '_' + str(state) + '_' + 'Live3' + '_1.txt']
    # path + str(date) + '/' + 'cam1' + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + 'Live2' + '_1794.txt']
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_108000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_162000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_216000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_270000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_324000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_378000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_432000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_486000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_540000.txt',\
    # path + str(date) + '/' + str(camera) + '/' + str(state) + '/' + str(date) + '_' + str(state) + '_' + str(camera) + '_594000.txt']
    
# Figure out how many jumps (or files) we are loading.          
n = len(jfile)

# Smoothing parameters, these can be modified base the quality of the recording
lam_big = 11 # for filtering baseline
lam_small = 10  # for filtering trace

# Frames per second
fps = 10
## Set the first and last indices of interest
data_start = 1
data_end = 100# end

###############################################################################
###############################################################################

# Set threshold for normalized activity trace to i.d. peaks
thresh = [0.1]

# Hours of recording
jumps = [0]

###############################################################################
###############################################################################

# Which jellyfish? (options: 0-7) 
jelly_numb = 1
jtype = 'Medusa'

###############################################################################
##############################################################################

# array of:
# [distance (time) between peaks, i.e. IPI for each jump (hour), \
#  distance (# frames) between peaks for each jump (hour)]
p_dist_lst = []

# array of frame number with pulse peak for each jump (hour)
p_t_lst = []

# array of number of peaks within time period (20 min) for each jump (hour)
p_numb_lst = []

# sum of all peaks, initialize to zero
p_numb_all = 0

# loop through each jump (hour) and append peak_counter properties
for itm in jumps:
    # Load data frame
    df = pd.read_csv(jfile[itm], comment='#', delim_whitespace=True, skiprows=2)
    
    # pull data from one jellyfish
    jelly = df.columns[jelly_numb]
    
    # calculate peak_counter properties for jellyfish for single jump (hour)
    p_dist, p_dist_t, p_numb, p_t = cb.peak_counter(df, jelly, thresh[itm], fps, lam_big, lam_small, data_start, data_end, itm)
        
    # p_dist, p_dist_t, p_numb, p_t = cb.peak_counter_2(df, jelly, thresh[itm], fps, lam_big, lam_small, 1795, 3589, itm)
        
    # append data from single jump (hour) for jellyfish 
    p_dist_lst.append(p_dist_t)
    p_dist_lst.append(p_dist)
    p_numb_lst.append(p_numb)
    p_t_lst.append(p_t)
    print('pulses for hour '+ str(itm) + ': ' + str(p_numb[0]))
    
    # update the total number of pulses for jellyfish
    p_numb_all = p_numb[0] + p_numb_all

# print total number of pulses for whole recording for jellyfish   
print('Total pulses for jelly # '+ str(jelly_numb))
print(p_numb_all)

# saving data directory
path_save = path[0:-4] + 'Analyzed/'+ str(date) + '/' + 'cam4' + '/' + 'smell_main' + '/' + '3' +'/'

# make two different df for jellyfish with different peak_counter properties
# and save for downstream plotting

# for item in range(len(p_t_lst)):
#     df = cb.df_maker_IPI_time_stamp(p_dist_lst, item)
#     df_2 = cb.df_maker_peak_time(p_t_lst, item)
#     df.to_csv(path_save +'J3' + str(jelly_numb) + '_' + str(jumps[item]) + '_dist.csv', index=False)
#     df_2.to_csv(path_save +'J3' + str(jelly_numb) + '_' + str(jumps[item]) + '_peak.csv', index=False)
