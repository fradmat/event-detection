import numpy as np
from keras.models import Model
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import os
import pickle
import sys
from cnn_data_generator import *
from cnn_model import *
# from keras.callbacks import 

class LossHistory(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        # self.train_size = train_size
    
    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch
        
    def on_train_begin(self, logs={}):
        self.elm_losses = []
        self.transition_losses = []
        self.elm_accuracies = []
        self.transition_accuracies = []
        self.epochs = []

    def on_batch_end(self, batch, logs={}):
        self.elm_losses.append(logs.get('out_elms_loss'))
        self.transition_losses.append(logs.get('out_transitions_loss'))
        self.elm_accuracies.append(logs.get('out_elms_categorical_accuracy'))
        self.transition_accuracies.append(logs.get('out_transitions_categorical_accuracy'))
        self.epochs.append(self.current_epoch)
        
    
    def on_train_end(self, logs={}):
        pass

def main():
    # print('HERE')
    stateful = False
    compress = True
    randomized_compression = False
    gaussian_time_window = 1e-3
    signal_sampling_rate = 1e4
    conv_w_size = 100
    # labelers = ['labit', 'ffelici']
    # labelers = ['ffelici']
    bsize = 8 * 8 #as we have 8 targets, each will have 4 samples in a batch
    augmented_per_sample = 3
    epsize = 32
    noeps = 2
    labelers = sys.argv[2].split(',')
    conv_offset = 20
    conv_channels = 4
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
    # exit(0)
    
    all_shots = [   61057,57103,26386,33459,43454,34010,32716,32191,61021,33638,
                    30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,
                    31650,31554,42514,26383,48580,62744,32794,30310,31211,31807,
                    47962,57751,31718,58460,57218,33188,56662,33271,30290,33567,
                    33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,42197,30262,45103,33446] #39872 42062
    # print(len(all_shots), len(all_shots)//2)
    random.shuffle(all_shots)
    train_shots = all_shots[:len(all_shots)//2]
    val_shots = all_shots[len(all_shots)//2:]
    print('randomized train shot ids', train_shots)
    print('randomized val shots', val_shots)
    assert len(set(train_shots) & set(val_shots)) == 0
    
    params_random_train = {
                'augmented': augmented_per_sample,
                'batch_size': bsize,
                'n_classes': 7,
                'shuffle': True,
                'epoch_size': epsize,
                'train_data_name': '',
                'no_input_channels' : conv_channels,
                'conv_w_size':conv_w_size,
                'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
                'labelers':labelers,
                'conv_offset':conv_offset,
                'shot_ids': train_shots}
    params_random_val = {
                'augmented': augmented_per_sample,
                'batch_size': bsize,
                'n_classes': 7,
                'shuffle': True,
                'epoch_size': epsize,
                'train_data_name': '',
                'no_input_channels' : conv_channels,
                'conv_w_size':conv_w_size,
                'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
                'labelers':labelers,
                'conv_offset':conv_offset,
                'shot_ids': val_shots}
    save_dic(params_random_train, train_dir + '/params_data_train')
    save_dic(params_random_val, train_dir + '/params_data_test')
    training_generator = CNNDataGenerator(**params_random_train)
    val_generator = CNNDataGenerator(**params_random_val)

    

    model = cnn(conv_w_size=conv_w_size, conv_channels=conv_channels)
    model.compile(optimizer='adam',
                  # loss={'out_elms': 'categorical_crossentropy', 'out_transitions': 'categorical_crossentropy', 'out_dithers': 'categorical_crossentropy'},
                  loss={'out_elms': 'categorical_crossentropy', 'out_transitions': 'categorical_crossentropy'},
                  # loss={'out_transitions': 'categorical_crossentropy'},
                  metrics = ['categorical_accuracy',]#keras example values, to change
                  # loss_weights={'main_output': 1., 'aux_output': 0.2}
                  )
    saveCheckpoint = ModelCheckpoint(filepath=checkpoint_dir + 'weights.weights.{epoch:02d}.h5', period=1)
    tb = TensorBoard(log_dir=logs_dir)
    model.fit_generator(generator=next(iter(training_generator)), steps_per_epoch=epsize, epochs=noeps,
                        verbose=1,callbacks=[tb, saveCheckpoint], validation_data = next(iter(val_generator)), validation_steps = 512) # 
    # print 'finished training.'
    
    model.save_weights(checkpoint_dir + 'weights.' + str(noeps) + ".h5")
    
    
if __name__ == '__main__':
    main()