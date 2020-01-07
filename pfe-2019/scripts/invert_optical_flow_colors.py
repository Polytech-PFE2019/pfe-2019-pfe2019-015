from __future__ import division

import numpy as np
import cupy
from joblib import Parallel, delayed
import multiprocessing
from skimage.transform import resize


import os
import scipy.misc
import matplotlib.pyplot as plt

def invert_frame(f):
	frame = plt.imread(f)
	frame = resize(frame, (480, 960, 3))
	return 255 - frame

def generate_flow(i):
	print("processing video: ", i)
	flow_path = '/data1/mohamed/Flow/flow/'+i
	output_path = '/data1/mohamed/Flow/'+i
	os.system('mkdir '+output_path)
	for frame in os.listdir(flow_path):
		scipy.misc.imsave(output_path+'/'+frame, invert_frame(flow_path+'/'+frame))
		
all_videos = os.listdir("/data1/mohamed/Flow/flow")
Parallel(n_jobs=20)(delayed(generate_flow)(i) for i in all_videos)