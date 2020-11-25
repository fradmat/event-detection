import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from lstm_data_generator import *
from lstm_model import *
from keras.utils import plot_model
from helper_funcs import get_date_time_formatted
import random


dtime = get_date_time_formatted()
train_dir = './experiments/' + sys.argv[1]
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
print('Will save this model to', train_dir)

checkpoint_dir = train_dir +'/model_checkpoints/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
logs_dir = train_dir +'/logs/'
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)
    
num_classes = 3
lstm_spread = int(200)
timesteps = None
conv_w_size = 40
epoch_size = 64
train_data_name = 'endtoendrandomizedshot' #elm wthresholdconvnetout 4hsots
no_input_channels = 4
bsize = 16*4
gaussian_time_window = 5e-4
signal_sampling_rate = 1e4
no_epocs = 500
stride = 5
conv_w_offset = 20
labelers = sys.argv[2].split(',')
shuffle=True
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
# tb = TensorBoard(log_dir=logs_dir)

all_shots = [   61057,57103,26386,33459,43454,34010,32716,32191,61021,33638,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,
                31650,31554,42514,26383,48580,62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,42197,
                33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,30262,45103,33446,33567] #39872 42062

random.shuffle(all_shots)
train_shots = all_shots[:len(all_shots)//2]
val_shots = all_shots[len(all_shots)//2:]

# train_shots = [53601, 47962, 61021, 31839, 33638, 31650, 31718, 45103, 32592, 30044, 33567, 26383, 52302, 32195, 26386, 59825, 33271, 56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103]
# val_shots = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105]
# train_shots = [32716,32191,32195,32911,32794,31211,47962,58182,32592,30262]

val_shots = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105]
train_shots=[53601, 47962, 61021, 31839, 33638, 31650, 31718, 45103, 32592, 30044, 33567, 26383, 52302, 32195, 26386, 59825, 33271, 56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103]


train_shots = []
train_shots_gino_apau=[64060, 60992, 60995, 64067, 61000, 61005, 61009, 61010, 61038, 61039, 61043, 64647, 64648,
             64658, 64659, 64662, 64666, 64670, 64675, 64678, 57000, 64680, 64686, 57009, 57010, 57011,
             59061, 57013, 59064, 59065, 59066, 61630, 61631, 57024, 59073, 57026, 59076, 57077, 57081,
             64770, 57093, 64774, 57095, 61702, 57094, 61703, 61711, 61712, 61713, 61714, 61716, 57622,
             57623, 57624, 61719, 64820, 61237, 61242, 61246, 61254, 64327, 61260, 64335, 64336, 64340,
             64342, 61274, 61275, 61279, 61281, 63843, 57706, 64363, 64364, 64369, 57715, 64371, 64373,
             53623, 64376, 53625, 53627, 53629, 57732, 60812, 60813, 60814, 60830]
train_shots.extend(train_shots_gino_apau)
# train_shots_ff_mau_ben = [61057,57103,26386,33459,43454,34010,32716,32191,61021,33638,30197,31839,60097,60275,
#                         32195,32911,59825,53601,34309,30268,31650,31554,42514,26383,48580,62744,32794,30310,
#                         31211,31807,47962,57751,31718,58460,57218,33188,56662,33271,30290,42197,33281,30225,
#                         58182,32592,30044,30043,29511,33942,45105,52302,30262,45103]
train_shots_ff_mau_ben =[
                        30268, 61057, 30290, 30197,
                         43454, 30310, 60097, 32794, 60275, 33942,33281, 42514, 62744, 30225,
                         29511, 34010, 31211, 34309, 32911, 31807,33459, 57218, 32191, 58460,
                         31554, 30262, 45105, 53601, 47962, 61021,31839, 33638, 31650, 31718,
                         45103, 32592, 30044,
                         33567, 26383, 52302,32195, 26386, 59825, 33271,
                         56662, 57751, 58182, 33188, 30043, 32716,42197, 33446, 48580, 57103]
train_shots.extend(train_shots_ff_mau_ben)
val_shots = []
val_shots_new = [59073, 61714, 61274, 59065, 61010, 61043, 64770, 64774, 64369, 64060, 64662, 64376, 57093, 57095,
           61021, 32911, 30268, 45105, 62744, 60097, 58460, 61057, 31807, 33459, 34309, 53601, 42197]
val_shots.extend(val_shots_new)
val_shots_old_paper = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514,
           62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105] #old_paper
# train_shots=[58182,]
# val_shots.extend(val_shots_old_paper)
# print(val_shots)
train_shots = set(train_shots)
val_shots = set(val_shots)
train_shots = list(train_shots - val_shots)
val_shots = list(val_shots)
print('randomized train shot ids', train_shots, len(train_shots))
print('randomized val shots ids', val_shots, len(val_shots))
    


print('randomized train shot ids', train_shots)
print('randomized val shots', val_shots)
# assert len(set(train_shots) & set(val_shots)) == 0
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
            'conv_w_offset':conv_w_offset,
            'machine_id':'TCV'}

print('experiment parameters', params_lstm_random)
params_random_train = {'shot_ids': train_shots}
params_random_train.update(params_lstm_random)
params_random_val = {'shot_ids': val_shots}
params_random_val.update(params_lstm_random)

save_dic(params_random_train, train_dir + '/params_data_train')
save_dic(params_random_val, train_dir + '/params_data_test')

training_generator = LSTMDataGenerator(**params_random_train)
gen_train = next(iter(training_generator))
val_generator = LSTMDataGenerator(**params_random_val)
gen_val = next(iter(val_generator))
# exit(0)
saveCheckpoint = ModelCheckpoint(filepath= checkpoint_dir + 'weights.{epoch:02d}.h5', period=1)

modelJoint.fit_generator(generator = gen_train, steps_per_epoch=epoch_size, epochs=no_epocs, validation_data=gen_val, validation_steps=bsize, callbacks=[saveCheckpoint,]) #,tb ,validation_data=gen_val, validation_steps=bsize

modelJoint.save_weights(checkpoint_dir + 'weights.' + str(no_epocs) + ".h5")
