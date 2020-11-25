import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from lstm_data_generator import *
from model import *
from keras.utils import plot_model
from helper_funcs import get_date_time_formatted
import random


dtime = get_date_time_formatted()
train_dir = sys.argv[1]
print('Will save this model to', train_dir)
ep_start = sys.argv[2]
no_epocs = int(ep_start) + 150
params_random_train = load_dic(train_dir + '/params_data_train')
params_random_val = load_dic(train_dir + '/params_data_test')
bsize = params_random_train['batch_size']
conv_w_size = params_random_train['conv_w_size']
no_input_channels = params_random_train['no_input_channels']
timesteps = None
epoch_size = params_random_train['epoch_size']
num_classes = params_random_train['no_classes']


if not os.path.isdir(train_dir + '/logs' + ep_start):
    os.makedirs(train_dir + '/logs' + ep_start)
# if not os.path.isdir(train_dir + '/model_checkpoints' + ep_start):
#     os.makedirs(train_dir + '/model_checkpoints' + ep_start)
# -------- Training --------
modelJoint = model_arc(bsize, conv_w_size, no_input_channels, timesteps, num_classes)
modelJoint.compile(loss={
                    'out_states':'categorical_crossentropy',
                    # 'out_transitions':'categorical_crossentropy',
                    'out_elms':'categorical_crossentropy'},
                   optimizer='adam',
                   metrics={
                    'out_states':'categorical_accuracy',
                    # 'out_transitions':'categorical_accuracy',
                    'out_elms':'categorical_accuracy'
                    }) # , loss_weights=[10,1]
tb = TensorBoard(log_dir=train_dir+'/logs_continuation')
modelJoint.load_weights('./' + train_dir + '/model_checkpoints/weights.' + ep_start + '.h5')


training_generator = LSTMDataGenerator(**params_random_train)
gen_train = next(iter(training_generator))
val_generator = LSTMDataGenerator(**params_random_val)
gen_val = next(iter(val_generator))
# print('here1')
saveCheckpoint = ModelCheckpoint(filepath=train_dir+'/model_checkpoints/weights.{epoch:02d}.h5', period=1)
# print('here2')
modelJoint.fit_generator(generator = gen_train, steps_per_epoch=epoch_size, epochs=no_epocs, callbacks=[tb,saveCheckpoint],validation_data=gen_val, validation_steps=bsize, initial_epoch=int(ep_start) + 1) #, 
# print('here3')

modelJoint.save_weights(train_dir + "/model_checkpoints/weights." + str(no_epocs + int(ep_start)) + ".h5")
