from DataGenerator import DataGenerator
from network import global_network
import numpy as np
import os
import keras
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


train_path1='/data1/mohamed/dataset/train/'
train_path2='/data3/mohamed_test/dataset/train/'
train_val_ratio = 0.9
# get the list of files storing data
all_files = []
delete_list = []

for frame in os.listdir(train_path1):
	# check if there are 5 consecutif frames in the dataset
	f = frame.split('_')
	second = "_".join([f[0], str(int(f[1])+1), f[2]])
	third = "_".join([f[0], str(int(f[1])+2), f[2]])
	fourth = "_".join([f[0], str(int(f[1])+3), f[2]])
	fifth = "_".join([f[0], str(int(f[1])+4), f[2]])
	if os.path.exists(train_path1+second) and os.path.exists(train_path1+third) and os.path.exists(train_path1+fourth) and os.path.exists(train_path1+fifth):
		all_files.append(frame)
		delete_list.append(second)
		delete_list.append(third)
		delete_list.append(fourth)
		delete_list.append(fifth)
			
for frame in os.listdir(train_path2):
	# check if there are 5 consecutive frames in the dataset
	f = frame.split('_')
	second = "_".join([f[0], str(int(f[1])+1), f[2]])
	third = "_".join([f[0], str(int(f[1])+2), f[2]])
	fourth = "_".join([f[0], str(int(f[1])+3), f[2]])
	fifth = "_".join([f[0], str(int(f[1])+4), f[2]])
	if os.path.exists(train_path2+second) and os.path.exists(train_path2+third) and os.path.exists(train_path2+fourth) and os.path.exists(train_path2+fifth):
		all_files.append(frame)	
		delete_list.append(second)
		delete_list.append(third)
		delete_list.append(fourth)
		delete_list.append(fifth)		



all_files = list(set(all_files) - set(delete_list))

split = int(len(all_files)*train_val_ratio)
train_files, valid_files = all_files[:split], all_files[split:]



# build the data generator
training_generator = DataGenerator(train_files, 5, training=True)
validation_generator = DataGenerator(valid_files, 5, training=True)
# build the model
network_model = global_network()
#parallel_model = global_network()
parallel_model = multi_gpu_model(network_model, gpus=4)

checkpointer = ModelCheckpoint(filepath='/data3/mohamed_test/model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='auto')
sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)
parallel_model.compile(optimizer=sgd, loss='mean_squared_error')
# start training
parallel_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=30,
                    callbacks=[checkpointer, earlystopping])			
