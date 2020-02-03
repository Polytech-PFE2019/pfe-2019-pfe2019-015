from DataGenerator import DataGenerator
from network import global_network
import numpy as np
import os
import keras
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


test_path='/data3/mohamed_test/dataset/test/'
# get the list of files storing data
all_files = []
delete_list = []

for frame in os.listdir(test_path):
	# check if there are 5 consecutif frames in the dataset
	f = frame.split('_')
	second = "_".join([f[0], str(int(f[1])+1), f[2]])
	third = "_".join([f[0], str(int(f[1])+2), f[2]])
	fourth = "_".join([f[0], str(int(f[1])+3), f[2]])
	fifth = "_".join([f[0], str(int(f[1])+4), f[2]])
	if os.path.exists(test_path+second) and os.path.exists(test_path+third) and os.path.exists(test_path+fourth) and os.path.exists(test_path+fifth):
		all_files.append(frame)
		delete_list.append(second)
		delete_list.append(third)
		delete_list.append(fourth)
		delete_list.append(fifth)
			

all_files = list(set(all_files) - set(delete_list))


# build the data generator
test_generator = DataGenerator(all_files, 2, training=False)
# build the model
network_model = global_network()
sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)
network_model.compile(optimizer=sgd, loss='mean_squared_error')
network_model.load_weights('/data3/mohamed_test/model.hdf5')

# start evaluation
loss = network_model.evaluate_generator(test_generator, verbose=1)			
print(loss)
