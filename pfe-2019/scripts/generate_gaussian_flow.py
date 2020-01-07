"""
 Get FOV gaussian mask from a frame according to a gaze position
"""
from __future__ import division

import numpy as np
import cupy
from matplotlib import pyplot as plt
import time
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import scipy.misc
import os

# a function to generate a gaussian mask when given the position of its center
def gauss_window(x_pos=480, y_pos=240):
  x, y = cupy.meshgrid(cupy.linspace(-x_pos,960-x_pos,960), cupy.linspace(480-y_pos,-y_pos,480))
  d = cupy.sqrt(x*x+y*y)
  # the FoV in a VR headset is about 100 degrees. This takes about 266 px horizontally since the panoramic scene is 360 degrees in 960 px.
  # Therefore we take a value of sigma that can cover this FoV
  sigma, mu = 85.0, 0
  channel = cupy.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
  return cupy.stack((channel,) * 3, axis=-1)

# a function to generate the masked frame given the full panoramic view of the frame and the gaze position
def masked_frame(frame, gaze_pos):
  # convert the gaze position to pixels in the panoramic frame
  x_frame = frame * cupy.array([255])
  x_pixeled = gaze_pos * cupy.array([960, 480])
  gaussian_mask = gauss_window(x_pixeled[0], x_pixeled[1])
  # Applying the gaussian mask to the frame
  result = cupy.multiply(gaussian_mask, x_frame)
  return result.astype('uint8')
  
def get_sampled_frames(gazes_history,f_i=5, step=5):
  if f_i - step + 1 < 0 or f_i + step + 1 > len(gazes_history[1]):
    return [], []
  # use the history gaze path in the first step frames to predict the gaze points in next step frames
  frames_X = np.array(gazes_history[1][f_i-step+1:f_i+1])
  frames_Y = np.array(gazes_history[1][f_i+1:f_i+step+1]) - frames_X
  return frames_X, frames_Y
  



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
      sampling_rate = len(trajectory_list) / len(os.listdir('/data1/mohamed/videos/'+video_name.split(".")[0]))
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
train_ids = pd.read_excel("train_test_set.xlsx",header=None, sheet_name='train_set').to_numpy().flatten()
test_ids = pd.read_excel("train_test_set.xlsx",header=None, sheet_name='test_set').to_numpy().flatten()  


def generate_gaussian_fov(videos_frames_list, i):
	# create a directory for gaussian masks for each video
	p_video = '{0:03d}'.format(i)
	gaussian_path = '/data1/mohamed/gaussian_flow/'+p_video+'/'
	flow_path = '/data1/mohamed/flow/'+p_video+'/'
	os.system('mkdir '+gaussian_path)
	print("processing video: ", i)
	video_all_viewers = videos_frames_list[videos_frames_list[:,0]==i]
	j = 0
	# a video can have many viewers
	for view in video_all_viewers:
	  # create a directory for patched frame for each viewer
	  viewer_path = gaussian_path+str(j)
	  os.system('mkdir '+viewer_path)
	  # get a gaze trace for each 5 frames
	  for frame in range(4, len(view[1])-5):
		sampled_frames = cupy.array(get_sampled_frames(view, frame))
		# generate the patched flow
		frame_name = '{0:06d}'.format(frame)
		scipy.misc.imsave(viewer_path+"/"+frame_name+".jpg", cupy.asnumpy(masked_frame(cupy.array(plt.imread(flow_path+frame_name+'.flo.png')),
												  sampled_frames[0][-1])))
	  j+=1
 
def generate_full_gaussian_fov(train_ids, test_ids):
	os.system('mkdir /data1/mohamed/gaussian_flow')
	# create gaussian masks for optical flows in videos of training
	Parallel(n_jobs=20)(delayed(generate_gaussian_fov)(videos_frames_list, i) for i in train_ids)
	# create gaussian masks for optical flows in videos of testing
	Parallel(n_jobs=20)(delayed(generate_gaussian_fov)(videos_frames_list, i) for i in test_ids)
  
generate_full_gaussian_fov(train_ids, test_ids)
