# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:40:55 2022

@author: wcab2
"""

import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


path = "C:/Users/wcab2/Documents/Jellyfish_data/yolo_test_save/Live_6/J2_detect_6.csv"
im = plt.imread('C:/Users/wcab2/Documents/Jellyfish_data/LIVE 6(2)/Frame (1).tiff')
im_crop_path = 'C:/Users/wcab2/Documents/Jellyfish_data/LIVE 6(2)/'
save_path = 'C:/Users/wcab2/Documents/Jellyfish_data/yolo_test_save/Live_6/crops/'


fig = plt.figure()

plt.xlim(0, 2700)
plt.ylim(2200,0)

distance = 400 #croping distance from center
frames = 5952

cor_lst = pd.read_csv(path, skiprows=None)
cor_array = []
cor_array.append(np.array(cor_lst))

color = range(0, frames)
    
x_mid = []
y_mid = []

xmin = []
ymin = []
xmax = []
ymax = []

for item in range(0,frames):
    xmin.append(float(cor_array[0][item][0]))
    ymin.append(float(cor_array[0][item][1]))
    xmax.append(float(cor_array[0][item][2]))
    ymax.append(float(cor_array[0][item][3]))

for i in range(0,frames):# can also use ymin, xmax, etc...
    x_mid.append((xmin[i] + xmax[i])/2)
    y_mid.append((ymin[i] + ymax[i])/2)
    
    
# for i in tqdm(range(1,frames)):
#     if x_mid[i] > 50:
#         pre_crop_im = Image.open(im_crop_path + 'Frame (' + str(i) + ')' + '.tiff')
#         # pre_crop_im = Image.open(im_crop_path + 'Frame ' + str(i) + '.tiff')
#         im_crop = pre_crop_im.crop((x_mid[i] - distance, y_mid[i] - distance ,x_mid[i] + distance, y_mid[i] + distance)) # (left,top,right,bottom))
#         im_crop.save(save_path + 'Frame_' + str(i) + '.jpg')
    

# plt.imshow(im)
plt.scatter(x_mid, y_mid, c = color, cmap='binary')
cbar = plt.colorbar(orientation = 'horizontal')
cbar.set_label(label = 'Time(Frames)')
plt.show()
