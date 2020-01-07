# create a dictionary of gazes per viewer
from __future__ import division

import numpy as np
import os
from sklearn import preprocessing
import random


# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = np.pi
        self.PI_2 = np.pi * 0.5
        self.PI2 = np.pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        _x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        x2 = np.mod(_x2, self.frame_width)
        y2 = np.minimum(y2, self.frame_height - 1)

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(_x2 - uf, y2 - vf)
        wb = np.multiply(_x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord), spericalCoord
		
		

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def backproject_to_equirectangular(equir_frame_, screen_cords_, values):
    uf = np.mod(screen_cords_.T[0], 1) * equir_frame_.shape[1]  # long - width
    vf = np.mod(screen_cords_.T[1], 1) * equir_frame_.shape[0]  # lat - height

    x0 = np.floor(uf).astype(int)  # coord of pixel
    y0 = np.floor(vf).astype(int)
    equir_frame_[:] = 1
    base_y0 = np.multiply(y0, equir_frame_.shape[1])
    A_idx = np.add(base_y0, x0)
    
    flat_img = np.copy(np.reshape(equir_frame_, [-1, equir_frame_.shape[2]]))
    flat_img[A_idx] = np.reshape(values, [-1, values.shape[2]])
    return flat_img


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


import matplotlib.pyplot as plt
import numpy as np

def get_patched_frame(frame, gaze_pos):
  nfov = NFOV()
  # (x, y) Between 0.0 and 1.0, origin in top-left
  #equir_frame = np.asarray(Image.open(frame))
  equir_frame = plt.imread(frame)
  fov_frame, screen_cords = nfov.toNFOV(equir_frame, center_point=np.array([gaze_pos[0], 1 - gaze_pos[1]]))
  #orig_fov_frame = np.copy(fov_frame)
  equir_frame_copy = np.copy(equir_frame)

  # backproject the FoV into the equirectangular frame
  flat_img = backproject_to_equirectangular(equir_frame_copy, screen_cords, fov_frame)
  return np.reshape(flat_img, equir_frame.shape)
  

# generate patched frames for the whole dataset
#from imageio import imsave
import scipy.misc

# generate the patched frames for all viewers and all videos
def generate_patched_frames(train_ids, test_ids, videos_frames_list):
  # create a directory for patched frames
  os.system('mkdir /data1/mohamed/patched')
  # create the patches of training
  for i in train_ids:
    # create a directory for patched frames for each video
    p_video = '{0:03d}'.format(i)
    os.system('mkdir /data1/mohamed/patched/'+p_video)
    print("processing video: ", p_video)
    video_all_viewers = videos_frames_list[videos_frames_list[:,0]==i]
    j = 0
    v = 0
    for view in video_all_viewers:
      v+=1
      # create a directory for patched frame for each viewer
      os.system('mkdir /data1/mohamed/patched/'+p_video+'/'+str(j))
      patched_path = '/data1/mohamed/patched/'+p_video+'/'+str(j)
      for frame in range(4, len(view[1])-5):
        sampled_frames = get_sampled_frames(view, frame)
        if frame == 4: print("hello ", len(videos_frames_list[videos_frames_list[:,0]==1][0][1]))
        frame_name = '{0:03d}'.format(frame)
        scipy.misc.imsave(patched_path+"/"+frame_name+".jpg", get_patched_frame('/data1/mohamed/Videos/'+p_video+'/'+frame_name+'.jpg',
                                                  sampled_frames[0][-1]))
      j+=1
  # create the patches of testing
  for i in test_ids:
    # create a directory for patched frames for each video
    p_video = '{0:03d}'.format(i)
    os.system('mkdir /data1/mohamed/patched/'+p_video)
    print("processing video: ", p_video,"....")
    video_all_viewers = videos_frames_list[videos_frames_list[:,0]==i]
    j = 0
    v = 0
    # a video can have many viewers
    for view in video_all_viewers:
      v+=1
      # create a directory for patched frame for each viewer
      os.system('mkdir /data1/mohamed/patched/'+p_video+'/'+str(j))
      patched_path = '/data1/mohamed/patched/'+p_video+'/'+str(j)
      # get a gaze trace for each 5 frames
      for frame in range(4, len(view[1])-5):
        sampled_frames = get_sampled_frames(view, frame)
        # generate the patched frame
        frame_name = '{0:03d}'.format(frame)
        scipy.misc.imsave(patched_path+"/"+frame_name+".jpg", get_patched_frame('/data1/mohamed/Videos/'+p_video+'/'+frame_name+'.jpg',
                                                  sampled_frames[0][-1]))
      j+=1
	  

generate_patched_frames(train_ids, test_ids, videos_frames_list)