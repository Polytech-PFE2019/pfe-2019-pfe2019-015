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
from __future__ import division

import numpy as np
import cupy
from joblib import Parallel, delayed
import multiprocessing

import os
from sklearn import preprocessing
import random
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt


class NFOV():
    def __init__(self, height=400, width=800):
        self.PI = cupy.pi
        
        self.PI_2 = cupy.pi * 0.5
        self.PI2 = cupy.pi * 2.0
        
        self.height = height
        self.width = width
        self.FOV = cupy.array([0.45, 0.45])
        cupy.cuda.Stream.null.synchronize()
        self.screen_points = self._get_screen_img()
        

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * cupy.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * cupy.array([self.PI, self.PI_2]) * (cupy.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = cupy.meshgrid(cupy.linspace(0, 1, self.width), cupy.linspace(0, 1, self.height))
        return cupy.transpose(cupy.stack([cupy.ravel(xx), cupy.ravel(yy)]))

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = cupy.transpose(convertedScreenCoord)[0]
        y = cupy.transpose(convertedScreenCoord)[1]

        rou = cupy.sqrt(x ** 2 + y ** 2)
        c = cupy.arctan(rou)
        sin_c = cupy.sin(c)
        cos_c = cupy.cos(c)

        lat = cupy.arcsin(cos_c * cupy.sin(self.cp[1]) + (y * sin_c * cupy.cos(self.cp[1])) / rou)
        lon = self.cp[0] + cupy.arctan2(x * sin_c, rou * cupy.cos(self.cp[1]) * cos_c - y * cupy.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5
        return cupy.transpose(cupy.stack([lon, lat]))
		
    def _bilinear_interpolation(self, screen_coord):
        uf = cupy.mod(cupy.transpose(screen_coord)[0],1) * self.frame_width  # long - width
        vf = cupy.mod(cupy.transpose(screen_coord)[1],1) * self.frame_height  # lat - height

        x0 = cupy.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = cupy.floor(vf).astype(int)
        _x2 = cupy.add(x0, cupy.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = cupy.add(y0, cupy.ones(vf.shape).astype(int))

        x2 = cupy.mod(_x2, self.frame_width)
        y2 = cupy.minimum(y2, self.frame_height - 1)

        base_y0 = cupy.multiply(y0, self.frame_width)
        base_y2 = cupy.multiply(y2, self.frame_width)
		

        A_idx = cupy.add(base_y0, x0)
        B_idx = cupy.add(base_y2, x0)
        C_idx = cupy.add(base_y0, x2)
        D_idx = cupy.add(base_y2, x2)
        flat_img = cupy.reshape(self.frame, [-1, self.frame_channel])
        A = cupy.take(flat_img, A_idx, axis=0)
        B = cupy.take(flat_img, B_idx, axis=0)
        C = cupy.take(flat_img, C_idx, axis=0)
        D = cupy.take(flat_img, D_idx, axis=0)

        wa = cupy.multiply(_x2 - uf, y2 - vf)
        wb = cupy.multiply(_x2 - uf, vf - y0)
        wc = cupy.multiply(uf - x0, y2 - vf)
        wd = cupy.multiply(uf - x0, vf - y0)

        # interpolate
        AA = cupy.multiply(A, cupy.transpose(cupy.stack([wa, wa, wa])))
        BB = cupy.multiply(B, cupy.transpose(cupy.stack([wb, wb, wb])))
        CC = cupy.multiply(C, cupy.transpose(cupy.stack([wc, wc, wc])))
        DD = cupy.multiply(D, cupy.transpose(cupy.stack([wd, wd, wd])))
        nfov = cupy.reshape(cupy.around(AA + BB + CC + DD).astype(cupy.uint8), [self.height, self.width, 3])
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
		
		
def backproject_to_equirectangular(equir_frame_, screen_cords_, values):
    uf = cupy.mod(cupy.transpose(screen_cords_)[0], 1) * equir_frame_.shape[1]  # long - width
    vf = cupy.mod(cupy.transpose(screen_cords_)[1], 1) * equir_frame_.shape[0]  # lat - height

    x0 = cupy.floor(uf).astype(int)  # coord of pixel
    y0 = cupy.floor(vf).astype(int)

    equir_frame_[:] = 0
    base_y0 = cupy.multiply(y0, equir_frame_.shape[1])
    A_idx = cupy.add(base_y0, x0)
    flat_img = cupy.copy(cupy.reshape(equir_frame_, [-1, equir_frame_.shape[2]]))

    flat_img[A_idx] = cupy.reshape(values, [-1, values.shape[2]])
    return flat_img
	
def get_patched_frame(frame, gaze_pos):
  nfov = NFOV()

  # (x, y) Between 0.0 and 1.0, origin in top-left
  equir_frame = cupy.array(plt.imread(frame))
  fov_frame, screen_cords = nfov.toNFOV(equir_frame, center_point=cupy.array([gaze_pos[0], 1 - gaze_pos[1]]))
  equir_frame_copy = cupy.copy(equir_frame)
  # backproject the FoV into the equirectangular frame
  flat_img = backproject_to_equirectangular(equir_frame_copy, screen_cords, fov_frame)
  return cupy.asnumpy(cupy.reshape(flat_img, equir_frame.shape))



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

def generate_patched_flow(videos_frames_list, i):
	# create a directory for patched frames for each video optical flow
	p_video = '{0:03d}'.format(i)
	patched_path = '/data1/mohamed/patched_flow_inverted/'+p_video+'/'
	flow_path = '/data1/mohamed/Flow/'+p_video+'/'
	os.system('mkdir '+patched_path)
	print("processing video: ", i)
	video_all_viewers = videos_frames_list[videos_frames_list[:,0]==i]
	j = 0
	# a video can have many viewers
	for view in video_all_viewers:
	  # create a directory for patched frame for each viewer
	  viewer_path = patched_path+str(j)
	  os.system('mkdir '+viewer_path)
	  # get a gaze trace for each 5 frames
	  for frame in range(4, len(view[1])-5):
		sampled_frames = get_sampled_frames(view, frame)
		# generate the patched frame
		frame_name = '{0:03d}'.format(frame)
		scipy.misc.imsave(viewer_path+"/"+frame_name+".jpg", get_patched_frame(flow_path+'{0:06d}'.format(frame)+'.flo.png',
												  sampled_frames[0][-1]))
	  j+=1

def generate_full_flow_patches(train_ids, test_ids):
	# create patched optical flow for frames in videos of training
	Parallel(n_jobs=20)(delayed(generate_patched_flow)(videos_frames_list, i) for i in train_ids)
	# create patched optical flow for frames in videos of testing
	Parallel(n_jobs=20)(delayed(generate_patched_flow)(videos_frames_list, i) for i in test_ids)
  
generate_full_flow_patches(train_ids, test_ids)
