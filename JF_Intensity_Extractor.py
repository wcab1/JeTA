
"""
Script for extracting changes in pixel intensity of individual jellyfish
over time.

Oiginal code by Claire Bedbrook & Ravi Nath

"""
from __future__ import division, absolute_import, \
                                    print_function, unicode_literals
                                    
import os
import numpy as np
import scipy.signal

import matplotlib
# matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
from matplotlib import cm
import skimage.io
import jb_utils as jb
import cv2
import pandas as pd

# A whole bunch of skimage stuff
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.restoration
import skimage.segmentation

################################################################################
#### path contains raw images, intensity measurments are written to path2  #####
################################################################################
# Input DATE, DAY or NIGHT, CAMERA, and file type
date = '20170611' 
time = 'Smell_main'

camera = 'Live3'
# ftype = '.tif'
ftype = '.tiff'
ROI = 'small'
__name__ == "__main__"

path = 'E:/JELLYFISH DATA/LIVE 3 New/'
# path = 'C:/Users/wcab2/Documents/Jellyfish_data/yolo_test_save/Live_6/crops_2/'
path2 = "C:/Users/wcab2/Documents/Jellyfish-master/example_data/TXT/20170611/cam1/Smell_Main/"

################################################################################

sl = 3588

jp1 = 1

#### list of slices ############################################################

slst = [[jp1, jp1 + sl]]

#### Jumping through slice list ################################################
mat = []
for t in range(len(slst)): 
    ############################################################################
    ####### JFC: Jellyfish Condos: top right and bottom left coordinates #######
    ############################(x1, x2, y1, y2#################################
    ################# We will need to update this each time ####################
    ############################################################################
    plt.close('all')
    JFC = []
    JFC2 = []
    
    # Normally 9 JFCs, 8 Condos & 1 Background
    for each in range(1):
        img_1 = skimage.io.imread(path + 'Frame  (' + str(slst[t][0]) + ')' + ftype) # use for new cam recordings
        img_2 = skimage.io.imread(path + 'Frame  (' + str(slst[t][1]) + ')' + ftype)
        
        # img_1 = skimage.io.imread(path + 'Frame_' + str(slst[t][0]) + ftype)
        # img_2 = skimage.io.imread(path + 'Frame_' + str(slst[t][1]) + ftype)
      
        skimage.io.imshow(img_1)
        plt.draw()
        plt.show()
        #### NOTE: timeout=0 prevents time out AND use delete/backspace to undo 
        ####       a bad click!
       
        x = plt.ginput(2, timeout=0)# 3 works better than 2 - no need to hit enter after a click
        print(x)
        JFC.append(x)
        # plt.ginput(2, timeout=0)
        ### Use pic to iterate because there is no Frame_0
        # print (plt.ginput(2, timeout=0))
    plt.close('all')
    JFC2 = []
    # x1y1 = (round(JFC[0][0][1]), round(JFC[0][0][0]))# y and x values for first click
    # x2y2 = (round(JFC[0][1][1]), round(JFC[0][1][0]))
            
    x1y1 = (int(JFC[0][0][0]), int(JFC[0][0][1]))# reversed
    x2y2 = (int(JFC[0][1][0]), int(JFC[0][1][1]))      
        
    JFC2.append(x1y1)  
    JFC2.append(x2y2) 
    print(JFC2)

    try:
        for i in range(slst[t][0], slst[t][1]):
            data = cv2.imread(path + 'Frame  (' + str(i) + ')' + ftype)
            # data = cv2.imread(path + 'Frame_' + str(i) + ftype)
            

##### This list will be filled with JFC mean pixel intensity measurments
            lst = []
            alst = []

    # Iterates through 12 JFC ROI 
            for item in JFC2:
           
                imean = np.mean(data[JFC2[1][1]:JFC2[0][1], JFC2[0][0]:JFC2[1][0]]) # WB

                iarea = -1*(JFC2[1][0] - JFC2[0][0]) * (JFC2[1][1] - JFC2[0][1])
             
                lst.append(imean)
                alst.append(iarea)
                
            mat.append(lst + alst)
            #print('pic:  ' + str(i))
    except:
        print('pass')
        pass

print('loop: ' + str(t))


################################################################################
######## Writing our matrix of ROI measurments for all frames to a file ########
################################################################################
if __name__ == "__main__":
    # Will create 4 files, each file will have our 'jump' info
    count = 0
    for t in range(len(slst)):
        # Creating File
        w = open(os.path.join(path2, \
        date + '_' + time  + '_' + camera + '_' + str(slst[t][0]) + '.txt'), 'w')
        
        w.write(date + ' ' + time  + ' ' + camera)
        w.write('\n' + ROI + '\n')
       
        # Adding Jellyfish Columns
        for column in range(int(len(mat[0])/2)):
            w.write('J' + str(column) +'\t' )

        # Adding Area Columns
        for column in range(int(len(mat[0])/2)):
            w.write('J' + str(column) +'area' +'\t' )    
        
        # Adding Values for Jellyfish
        for row in mat[count: count + sl]:
            w.write('\n')
            for num in row:
                w.write('%.1f\t' % num)
        
                
        count += sl
        
        w.close()