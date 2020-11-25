from __future__ import print_function
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
import horovod.keras as hvd
from keras import backend as K

import tensorflow as tf

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

dtime = get_date_time_formatted()
train_dir = 'train-' + sys.argv[1]
train_dir = sys.argv[1]
print('Will save this model to', train_dir)


# no_workers = argv[3]
num_classes = 3
lstm_spread = int(200)
timesteps = None
conv_w_size = 40
epoch_size = 64
train_data_name = 'endtoendrandomizedshot' #elm wthresholdconvnetout 4hsots
no_input_channels = 4
bsize = 16*4*hvd.size()
gaussian_time_window = 10e-4
signal_sampling_rate = 1e4
no_epocs = 300
stride = 5
conv_w_offset = 20
labelers = sys.argv[2].split(',')
shuffle=True
# -------- Training --------

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.Adam(lr=0.001 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

modelJoint = model_arc(bsize, conv_w_size, no_input_channels, timesteps, num_classes)
modelJoint.compile(loss={
                    'out_states':'categorical_crossentropy',
                    # 'out_transitions':'categorical_crossentropy',
                    'out_elms':'categorical_crossentropy'},
                   # optimizer='adam',
                   optimizer=opt,
                   metrics={
                    'out_states':'categorical_accuracy',
                    # 'out_transitions':'categorical_accuracy',
                    'out_elms':'categorical_accuracy'
                    }) # , loss_weights=[10,1]


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.TensorBoard(log_dir=train_dir+'/logs'))
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=train_dir+'/model_checkpoints/weights.{epoch:02d}.h5', period=1))

all_shots = [61057,57103,26386,33459,43454,34010,32716,32191,61021,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
                31650,31554,42514,26383,48580,62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,
                33281,30225,58182,32592, 30044,30043,29511,33942,45105,52302,42197,30262,45103,33446,33567] #39872 42062
# print(len(all_shots), len(all_shots)//2)
random.shuffle(all_shots)
train_shots = all_shots[:len(all_shots)//2]
val_shots = all_shots[len(all_shots)//2:]
# train_shots = [61057,]
# val_shots = [57103,]
print('randomized train shot ids', train_shots)
print('randomized val shots', val_shots)
assert len(set(train_shots) & set(val_shots)) == 0
params_lstm_random = {
            'batch_size': int(bsize),
            'n_classes': 7,
            'lstm_time_spread': int(lstm_spread),
            'epoch_size': epoch_size,
            'train_data_name':train_data_name,
            'no_input_channels' : no_input_channels,
            'conv_w_size':conv_w_size,
            'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
            'no_classes': num_classes,
            'stride':int(stride),
            'labelers':labelers,
            'shuffle':shuffle,
            'conv_w_offset':conv_w_offset}

print('experiment parameters', params_lstm_random)
params_random_train = {'shot_ids': train_shots}
params_random_train.update(params_lstm_random)
params_random_val = {'shot_ids': val_shots}
params_random_val.update(params_lstm_random)

save_dic(params_random_train, train_dir + '/params_data_train')
save_dic(params_random_val, train_dir + '/params_data_val')

training_generator = LSTMDataGenerator(**params_random_train)
gen_train = next(iter(training_generator))
val_generator = LSTMDataGenerator(**params_random_val)
gen_val = next(iter(val_generator))


modelJoint.fit_generator(generator = gen_train, steps_per_epoch=epoch_size, epochs=no_epocs, callbacks=[tb,saveCheckpoint],validation_data=gen_val, validation_steps=bsize) #, 

modelJoint.save_weights(train_dir + "/model_checkpoints/weights." + str(no_epocs) + ".h5")
