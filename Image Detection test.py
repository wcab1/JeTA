# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:36:58 2022

@author: wcab2
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import pandas as pd
import time
from tqdm import tqdm

number_of_jellyfish = 2
frames = 5958
st_a = time.time()
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/wcab2/yolov5/jellyfish.pt', device = 'cpu' , _verbose=False)
# Images
im_path = 'C:/Users/wcab2/Documents/Jellyfish_data/LIVE 6(2)/'  # or file, Path, URL, PIL, OpenCV, numpy, list
# im_path = 'C:/Users/wcab2/yolov5/data/images/'    

save_path = open("C:/Users/wcab2/Documents/Jellyfish_data/yolo_test_save/Live_6/J_detect_6.csv", 'w', encoding='utf8', newline='')
save_path_2 = open("C:/Users/wcab2/Documents/Jellyfish_data/yolo_test_save/Live_6/J2_detect_6.csv", 'w', encoding='utf8', newline='')
crop_path = 'C:/Users/wcab2/yolov5/Code(WCAB)/Test_crops/'
cor =[]
writer = csv.writer(save_path)
writer2 = csv.writer(save_path_2)
headers = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'type']

st_a = time.time()

for i in tqdm(range(1,frames)):
    im = cv2.imread(im_path + 'Frame ' + '(' + str(i) + ')' + '.tiff')
    results = model(im)
    cor.append(results.pandas().xyxy[0])
    # crops = results.crop(save=True, exist_ok=True)
et_a = time.time()

st_w = time.time()

np_cor = []
for item in range(0,frames-1):
    np_cor.append(np.array(cor[item]))

if number_of_jellyfish==2:
    writer.writerow(headers)
    writer2.writerow(headers)
    for i in range(0, frames-1):
        try:
            if np_cor[i][0][0] > 1000:
                writer.writerow(np_cor[i][0])
            if np_cor[i][0][0] < 1000:
                writer2.writerow(np_cor[i][0])
            
            if np_cor[i][1][0] > 1000:
                writer.writerow(np_cor[i][1])
            if np_cor[i][1][0] < 1000:
                writer2.writerow(np_cor[i][1])
            # print(i)
        except IndexError as e:
            print(e)
   
    save_path.close()
    save_path_2.close()
else:
    writer.writerow(headers)
    for i in range(0,frames-1):
        writer.writerow(np_cor[i][0])
    save_path.close()
   
et_w = time.time()

elapsed_analysis = (et_a - st_a)
elapsed_write = et_w - st_w

if elapsed_analysis >= 60:
    print('Analysis time:', int(elapsed_analysis/60), 'minutes')
else:
    print('Analysis time:', (elapsed_analysis), 'seconds')

