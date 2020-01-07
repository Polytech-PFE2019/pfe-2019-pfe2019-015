from __future__ import division
import matplotlib.pyplot as plt
import sys
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import os
import cupy


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
	
	
train_save_path = '/data1/mohamed/dataset/train/'
test_save_path = '/data1/mohamed/dataset/test/'

def normalize(array):
	return cupy.array(array)/255
	

def generate_dataset(videos_frames_list, i, split):
	flow_path = '/data1/mohamed/flow/'+'{0:03d}'.format(i)
	patched_path = '/data1/mohamed/patched/{0:03d}'.format(i)
	panoramic_path = '/data1/mohamed/videos/{0:03d}'.format(i)
	gaussian_path = '/data1/mohamed/gaussian/{0:03d}'.format(i)
	patched_flow_path = '/data1/mohamed/patched_flow/{0:03d}'.format(i)
	saliency_path = '/data1/mohamed/saliency/{0:03d}'.format(i)
	patched_saliency = '/data1/mohamed/patched_saliency/{0:03d}'.format(i)
	gaussain_flow_path = '/data1/mohamed/gaussian_flow/{0:03d}'.format(i)
	
	video_all_viewers = videos_frames_list[videos_frames_list[:,0]==i]
	j = 0
	for view in video_all_viewers:
	  print("processing video:", i, "viewer", j)
	  for frame in range(4, len(view[1])-6):
		filename = str(i)+'_'+str(frame)+'_'+str(j)
		if not os.path.exists(train_save_path+filename+'.npz'):
			sampled_frames = get_sampled_frames(view, frame)
			# get the panoramic image at frame t+1
			panorama_t1 = normalize(plt.imread(panoramic_path+'/{0:03d}'.format(frame+1)+'.jpg'))
			# get the saliency map of the panoramic image at frame t+1
			sal_panorama_t1 = normalize(plt.imread(saliency_path+'/{0:03d}'.format(frame+1)+'.jpg'))
			# get the gaussian masked frame according to the gaze at frame t+1
			gauss_panorama_t1 = normalize(plt.imread(gaussian_path+'/'+str(j)+'/{0:03d}'.format(frame+1)+'.jpg'))
			# get the gaussian masked optical flow of the panoramic view between t and t+1
			gauss_flow_panorama_t1 = normalize(plt.imread(gaussain_flow_path+'/'+str(j)+'/{0:06d}'.format(frame+1)+'.jpg'))
			# get the optical flow of the panoramic view between t and t+1
			flow_panorama_t1 = cupy.array(plt.imread(flow_path+'/{0:06d}'.format(frame)+'.flo.png'))
			# get the patched optical flow of the frame according to the gaze at frame t+1
			patched_flow_panorama_t1 = normalize(plt.imread(patched_flow_path+'/'+str(j)+'/{0:03d}'.format(frame+1)+'.jpg'))
			# get the patched saliency of the frame according to the gaze at frame t+1
			patched_sal_panorama_t1 = normalize(plt.imread(patched_saliency+'/'+str(j)+'/{0:03d}'.format(frame+1)+'.jpg'))
			# get the patched frame according to the gaze at frame t+1
			patched_panorama_t1 = normalize(plt.imread(patched_path+'/'+str(j)+'/{0:03d}'.format(frame+1)+'.jpg'))
			if split == 0:
				cupy.savez_compressed(train_save_path+filename+'.npz',saliency=cupy.stack([panorama_t1,
				sal_panorama_t1,
				gauss_panorama_t1,
				gauss_flow_panorama_t1,
				flow_panorama_t1,
				patched_flow_panorama_t1,
				patched_sal_panorama_t1,
				patched_panorama_t1]), trajectory_x=sampled_frames[0], trajectory_y=sampled_frames[1])
			else:
				cupy.savez_compressed(test_save_path+filename+'.npz',saliency=cupy.stack([panorama_t1,
				sal_panorama_t1,
				gauss_panorama_t1,
				gauss_flow_panorama_t1,
				flow_panorama_t1,
				patched_flow_panorama_t1,
				patched_sal_panorama_t1,
				patched_panorama_t1]), trajectory_x=sampled_frames[0], trajectory_y=sampled_frames[1])
          cupy.cuda.Stream.null.synchronize()
	  j+=1
	
def generate_full_dataset(train_ids, test_ids, videos_frames_list):
  # create the dataset of training
  Parallel(n_jobs=20)(delayed(generate_dataset)(videos_frames_list, i, 0) for i in train_ids)
  # create the dataset of testing
  Parallel(n_jobs=20)(delayed(generate_dataset)(videos_frames_list, i, 1) for i in test_ids)

generate_full_dataset(train_ids, test_ids, videos_frames_list)
