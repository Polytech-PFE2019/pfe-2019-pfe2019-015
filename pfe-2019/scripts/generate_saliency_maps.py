from __future__ import division


import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py
import sys
sys.path.append('/data1/mohamed')
sys.path.append('/data1/mohamed/saliency_cvpr')
sys.path.append('/data1/mohamed/saliency_cvpr/deep')

from joblib import Parallel, delayed
import multiprocessing

from saliency_cvpr.deep.get_saliency import get_saliency_for_deepnet


import numpy as np
import os
from sklearn import preprocessing
import random

def get_sampled_frames(gazes_history,f_i=5, step=5):
  if f_i - step + 1 < 0 or f_i + step + 1 > len(gazes_history[1]):
    return [], []
  # use the history gaze path in the first step frames to predict the gaze points in next step frames
  frames_X = np.array(gazes_history[1][f_i-step+1:f_i+1])
  frames_Y = np.array(gazes_history[1][f_i+1:f_i+step+1]) - frames_X
  return frames_X, frames_Y
  
"""
 Get FOV gaussian mask from a frame according to a gaze position
"""
import numpy as np
from matplotlib import pyplot as plt
# a function to generate a gaussian mask when given the position of its center
def gauss_window(x_pos=480, y_pos=240):
  x, y = np.meshgrid(np.linspace(-x_pos,960-x_pos,960), np.linspace(480-y_pos,-y_pos,480))
  d = np.sqrt(x*x+y*y)
  # the FoV in a VR headset is about 100 degrees. This takes about 266 px horizontally since the panoramic scene is 360 degrees in 960 px.
  # Therefore we take a value of sigma that can cover this FoV
  sigma, mu = 85.0, 0
  channel = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
  return np.stack((channel,) * 3, axis=-1)

# a function to generate the masked frame given the full panoramic view of the frame and the gaze position
def masked_frame(frame, gaze_pos):
  # convert the gaze position to pixels in the panoramic frame
  x_pixeled = gaze_pos * [960, 480]
  gaussian_mask = gauss_window(gaze_pos[0], gaze_pos[1])
  # Applying the gaussian mask to the frame
  result = np.multiply(gaussian_mask, frame)
  return result.astype('uint8')
  
  

raw_data_path = "Gaze_txt_files"
dir_list = sorted(os.listdir(raw_data_path))
p_num = len(dir_list)
p_data_list = [{} for _ in range(p_num)]
# fill p_data
for dir_name in dir_list:
  person_gaze_txt_path = os.path.join(raw_data_path, dir_name)
  pid = int(dir_name.replace('p', ''))
  for video_name in os.listdir(person_gaze_txt_path):
    with open(os.path.join(person_gaze_txt_path,video_name), 'r') as f:
      trajectory_list = f.read().split()
      gaze_video_list = []
      f = 0
      sampling_rate = len(trajectory_list) / len(os.listdir('/data1/mohamed/Videos/'+video_name.split(".")[0]))
      for i in trajectory_list:
        # add only gazes for the sampled frames
        if f%sampling_rate<1:
          i = i.split(",")
          x = float(i[-2])
          y = float(i[-1])
          gaze_video_list.append((x,y))
        f+=1
      p_data_list[int(pid)-1][int(video_name.split(".")[0])] = np.array(gaze_video_list)
	  
	  
	  
# get all videos's gazes and store them in a list
videos_frames_list = np.array([np.array([j, p_data_list[i][j]]) for i in range(len(p_data_list)) for j in p_data_list[i].keys()])

# load train_test videos indices from the excel file 
import pandas as pd
train_ids = pd.read_excel("train_test_set.xlsx",header=None, sheet_name='train_set').to_numpy().flatten()
test_ids = pd.read_excel("train_test_set.xlsx",header=None, sheet_name='test_set').to_numpy().flatten()

import scipy.misc

def generate_saliency_frames(videos_frames_list, i):
	print("processing video:", i)
	panoramic_path = '/data1/mohamed/Videos/'+'{0:03d}'.format(i)
	saliency_path = '/data1/mohamed/saliency/'+'{0:03d}'.format(i)
	os.system('mkdir '+saliency_path)
	for frame in range(1, len(os.listdir(panoramic_path))+1):
		panorama_t = panoramic_path+'/'+'{0:03d}'.format(frame)
		# get the saliency map of the panoramic image at frame t
		sal_panorama_t = np.stack((get_saliency_for_deepnet(panorama_t),) * 3, axis=-1)
		scipy.misc.imsave(saliency_path+"/"+'{0:03d}'.format(frame)+".jpg", sal_panorama_t)
	
def generate_full_saliency_frames(train_ids, test_ids):
	os.system('mkdir /data1/mohamed/saliency')
	# create saliency maps for frames in videos of training
	Parallel(n_jobs=20)(delayed(generate_saliency_frames)(videos_frames_list, i) for i in train_ids)
	# create saliency maps for frames in videos of testing
	Parallel(n_jobs=20)(delayed(generate_saliency_frames)(videos_frames_list, i) for i in test_ids)
  
generate_full_saliency_frames(train_ids, test_ids)
