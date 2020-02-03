from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, concatenate, Reshape, Add
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K


def global_network():
	"""
	The trajectory encoder module
	"""
	# Define an input sequence of previous gazes and process it.
	encoder_inputs = Input(shape=(5, 2))
	# Define the first lstm layer
	lstm1 = LSTM(128, return_sequences=True, return_state=True)
	# Define the second lstm layer
	lstm2 = LSTM(128)

	"""
	The Saliency encoder module
	"""
	# Define input for all spatial and temporal saliency maps and process it.
	saliency_inputs = Input(shape=(5, 480*8, 960, 3))
	# get  Inception-ResNet-V2 to extract saliency features for gaze prediction followed with a global pooling
	inception = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(480*8, 960, 3), pooling='avg')
	for layer in inception.layers:
		layer.trainable = False
	# Two fully connected layer will be used to estimate the displacement
    # between the gaze point at time t + 1 and gaze point at time t
	dense_1 = Dense(1000)
	dense_2 = Dense(2)
	#saliency_encoder_outputs
	all_displacement_outputs=[]
	
	# First frame
	# apply the first lstm on the trajectory history of the 5 gazes
	lstm1_outputs_0, state_h_0, state_c_0 = lstm1(encoder_inputs)
	lstm1_states_0 = [state_h_0, state_c_0]
	# Set up the second lstm, using 'lstm1_states' as initial state.
	trajectory_encoder_outputs_0 = lstm2(lstm1_outputs_0, initial_state=lstm1_states_0)
	# get only the current saliency input at frame t
	saliency_input_0 = Lambda(lambda x: x[:,0])(saliency_inputs)
	# apply inception on the current saliency input
	saliency_inception_0 = inception(saliency_input_0)
	"""
	Displacement Prediction Module
	"""
	# Concatenate s the output of saliency encoder module and trajectory encoder module
	merged_0 = concatenate([saliency_inception_0, trajectory_encoder_outputs_0])
	# use two fully connected layer to estimate the displacement between the gaze point at time t + 1 and gaze point at time t
	dense_1000_0 = dense_1(merged_0)
	displacement_output_0 = dense_2(dense_1000_0)
	# get the current gaze positions at frame t to add to the predicted gaze displacement
	current_gaze_t_0 = Lambda(lambda x: x[:,-1])(encoder_inputs)
	traj_inputs_1 = concatenate([encoder_inputs, Reshape((1,2))(Add()([current_gaze_t_0, displacement_output_0]))], 1)
	# append the current predicted gaze point at t+1 to the list of 5 predicted outputs
	all_displacement_outputs.append(displacement_output_0)
	
	# Second frame
	# apply the first lstm on the trajectory history of the 5 gazes
	lstm1_outputs_1, state_h_1, state_c_1 = lstm1(traj_inputs_1)
	lstm1_states_1 = [state_h_1, state_c_1]
	# Set up the second lstm, using 'lstm1_states' as initial state.
	trajectory_encoder_outputs_1 = lstm2(lstm1_outputs_1, initial_state=lstm1_states_1)
	# get only the current saliency input at frame t
	saliency_input_1 = Lambda(lambda x: x[:,1])(saliency_inputs)
	# apply inception on the current saliency input
	saliency_inception_1 = inception(saliency_input_1)
	"""
	Displacement Prediction Module
	"""
	# Concatenate s the output of saliency encoder module and trajectory encoder module
	merged_1 = concatenate([saliency_inception_1, trajectory_encoder_outputs_1])
	# use two fully connected layer to estimate the displacement between the gaze point at time t + 1 and gaze point at time t
	dense_1000_1 = dense_1(merged_1)
	displacement_output_1 = dense_2(dense_1000_1)
	# get the current gaze positions at frame t to add to the predicted gaze displacement
	current_gaze_t_1 = Lambda(lambda x: x[:,-1])(traj_inputs_1)
	traj_inputs_2 = concatenate([traj_inputs_1, Reshape((1,2))(Add()([current_gaze_t_1, displacement_output_1]))], 1)
	# append the current predicted gaze point at t+1 to the list of 5 predicted outputs
	all_displacement_outputs.append(displacement_output_1)
	
	
	# Third frame
	# apply the first lstm on the trajectory history of the 5 gazes
	lstm1_outputs_2, state_h_2, state_c_2 = lstm1(traj_inputs_2)
	lstm1_states_2 = [state_h_2, state_c_2]
	# Set up the second lstm, using 'lstm1_states' as initial state.
	trajectory_encoder_outputs_2 = lstm2(lstm1_outputs_2, initial_state=lstm1_states_2)
	# get only the current saliency input at frame t
	saliency_input_2 = Lambda(lambda x: x[:,2])(saliency_inputs)
	# apply inception on the current saliency input
	saliency_inception_2 = inception(saliency_input_2)
	"""
	Displacement Prediction Module
	"""
	# Concatenate s the output of saliency encoder module and trajectory encoder module
	merged_2 = concatenate([saliency_inception_2, trajectory_encoder_outputs_2])
	# use two fully connected layer to estimate the displacement between the gaze point at time t + 1 and gaze point at time t
	dense_1000_2 = dense_1(merged_2)
	displacement_output_2 = dense_2(dense_1000_2)
	# get the current gaze positions at frame t to add to the predicted gaze displacement
	current_gaze_t_2 = Lambda(lambda x: x[:,-1])(traj_inputs_2)
	traj_inputs_3 = concatenate([traj_inputs_2, Reshape((1,2))(Add()([current_gaze_t_2, displacement_output_2]))], 1)
	# append the current predicted gaze point at t+1 to the list of 5 predicted outputs
	all_displacement_outputs.append(displacement_output_2)
	
	
	# Fourth frame
	# apply the first lstm on the trajectory history of the 5 gazes
	lstm1_outputs_3, state_h_3, state_c_3 = lstm1(traj_inputs_3)
	lstm1_states_3 = [state_h_3, state_c_3]
	# Set up the second lstm, using 'lstm1_states' as initial state.
	trajectory_encoder_outputs_3 = lstm2(lstm1_outputs_3, initial_state=lstm1_states_3)
	# get only the current saliency input at frame t
	saliency_input_3 = Lambda(lambda x: x[:,3])(saliency_inputs)
	# apply inception on the current saliency input
	saliency_inception_3 = inception(saliency_input_3)
	"""
	Displacement Prediction Module
	"""
	# Concatenate s the output of saliency encoder module and trajectory encoder module
	merged_3 = concatenate([saliency_inception_3, trajectory_encoder_outputs_3])
	# use two fully connected layer to estimate the displacement between the gaze point at time t + 1 and gaze point at time t
	dense_1000_3 = dense_1(merged_3)
	displacement_output_3 = dense_2(dense_1000_3)
	# get the current gaze positions at frame t to add to the predicted gaze displacement
	current_gaze_t_3 = Lambda(lambda x: x[:,-1])(traj_inputs_3)
	traj_inputs_4 = concatenate([traj_inputs_3, Reshape((1,2))(Add()([current_gaze_t_3, displacement_output_3]))], 1)
	# append the current predicted gaze point at t+1 to the list of 5 predicted outputs
	all_displacement_outputs.append(displacement_output_3)
	
	# Fifth frame
	# apply the first lstm on the trajectory history of the 5 gazes
	lstm1_outputs_4, state_h_4, state_c_4 = lstm1(traj_inputs_4)
	lstm1_states_4 = [state_h_4, state_c_4]
	# Set up the second lstm, using 'lstm1_states' as initial state.
	trajectory_encoder_outputs_4 = lstm2(lstm1_outputs_4, initial_state=lstm1_states_4)
	# get only the current saliency input at frame t
	saliency_input_4 = Lambda(lambda x: x[:,4])(saliency_inputs)
	# apply inception on the current saliency input
	saliency_inception_4 = inception(saliency_input_4)
	"""
	Displacement Prediction Module
	"""
	# Concatenate s the output of saliency encoder module and trajectory encoder module
	merged_4 = concatenate([saliency_inception_4, trajectory_encoder_outputs_4])
	# use two fully connected layer to estimate the displacement between the gaze point at time t + 1 and gaze point at time t
	dense_1000_4 = dense_1(merged_4)
	displacement_output_4 = dense_2(dense_1000_4)
	# append the current predicted gaze point at t+1 to the list of 5 predicted outputs
	all_displacement_outputs.append(displacement_output_4)
	
	# concatenate all the element of the output list to get the final output
	displacement_outputs = Reshape((5,2))(concatenate(all_displacement_outputs,1))
	# Create the proposed model
	model = Model([encoder_inputs, saliency_inputs], displacement_outputs)
	return model