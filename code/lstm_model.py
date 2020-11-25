from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from keras.layers import Input, MaxPooling1D, Flatten, concatenate, Dense, Dropout, BatchNormalization, Conv1D, Activation
from keras.models import Model

def model_arc(bsize, conv_w_size, no_input_channels, timesteps, num_classes):
    conv_input = Input(shape=(conv_w_size,no_input_channels,), dtype='float32', name='conv_input')
    
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(conv_input)
    x_conv = Conv1D(128, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv)
    x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    
    
    conv_out = Flatten()(x_conv)
    conv_out = Dense(64, activation='relu', name='scalar_output')(conv_out)
    conv_out = Dense(16, activation='relu')(conv_out)
    modelCNN = Model(inputs=[conv_input], outputs= [conv_out])
    print(modelCNN.summary())
    
    joint_input = Input(batch_shape=(int(bsize),timesteps,conv_w_size,no_input_channels), dtype='float32', name='in_scalars')
    # start_seq_input = Input(batch_shape=(int(bsize),1), dtype='float32', name='in_seq_start')
    # concat_inputs = concatenate([joint_input, start_seq_input])
    modelJoined = TimeDistributed(modelCNN)(joint_input)
    modelJoined = LSTM(32, return_sequences=True, stateful=False)(modelJoined)
    modelJoined = LSTM(32, return_sequences=True, stateful=False)(modelJoined)
    # modelJoined = Bidirectional(LSTM(32, return_sequences=True, stateful=False))(modelJoined)
    # modelJoined = Bidirectional(LSTM(32, return_sequences=True, stateful=False))(modelJoined)
    modelJoined = TimeDistributed(Dense(32, activation='relu'))(modelJoined)
    modelJoined = Dropout(.5)(modelJoined)
    modelJoinedStates = TimeDistributed(Dense(3, activation='softmax'),name='out_states')(modelJoined)
    # modelJoinedTransitions = TimeDistributed(Dense(7, activation='softmax'), name='out_transitions')(modelJoined)
    modelJoinedElms = TimeDistributed(Dense(2, activation='softmax'), name='out_elms')(modelJoined)
    modelJoint = Model(inputs=[joint_input], outputs= [modelJoinedStates,modelJoinedElms])
    # modelJoint = Model(inputs=[joint_input], outputs= [modelJoinedTransitions,modelJoinedElms])
    print(modelJoint.summary())
    return modelJoint
