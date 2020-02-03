import numpy as np
import keras
import os

train_path1='/data1/mohamed/dataset/train/'
train_path2='/data3/mohamed_test/dataset/train/'
test_path='/data3/mohamed_test/dataset/test/'


		
		
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_id, batch_size=8, shuffle=True, training=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_id
        self.shuffle = shuffle
        self.training = training
        self.on_epoch_end()
		
    def get_Nextframes(self, ID):
		files = []
		if self.training==True:
			for i in range(0,5):
				f = ID.split('_')
				frame = "_".join([f[0], str(int(f[1])+i), f[2]])
				if os.path.exists(train_path1+frame):
					files.append(train_path1+frame)
				elif os.path.exists(train_path2+frame):
					frame = "_".join([f[0], str(int(f[1])+i), f[2]])
					files.append(train_path2+frame)
				else:
					print(frame, "not found at index:", i)
		else:
			for i in range(0,5):
				f = ID.split('_')
				frame = "_".join([f[0], str(int(f[1])+i), f[2]])
				if os.path.exists(test_path+frame):
					files.append(test_path+frame)
				else:
					print(frame, "not found at index:", i)
		return files
	

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        saliency, trajectory_x, trajectory_y = self.__data_generation(list_IDs_temp)
        return [trajectory_x, saliency], trajectory_y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
			
	
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        saliency = np.empty((self.batch_size, 5, 480*8, 960, 3))
        trajectory_x = np.empty((self.batch_size, 5, 2))
        trajectory_y = np.empty((self.batch_size, 5, 2))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample saliency
            files = self.get_Nextframes(ID)
            try:
				current_frame = np.load(files[0], allow_pickle=True)
				saliency[i,] = np.array([np.concatenate(current_frame['saliency'],0),np.concatenate(np.load(files[1])['saliency'],0),np.concatenate(np.load(files[2])['saliency'],0),np.concatenate(np.load(files[3])['saliency'],0),np.concatenate(np.load(files[4])['saliency'],0)])
            except:
			    print(files)
			# Store sample trajectory
            trajectory_x[i,] = current_frame['trajectory_x']
            # Store trajectory prediction
            trajectory_y[i,] = current_frame['trajectory_y']
	return saliency, trajectory_x, trajectory_y
