from sequence_to_sequence_data_generator import *
from helper_funcs import load_fshot_from_labeler, normalize_signals_mean
from keras import backend as K
import os
import matplotlib as mpl
import tensorflow as tf
from sequence_to_sequence_funcs import train, predict
mpl.use('Agg')

def main(args):
    train_dir = './experiments/' + sys.argv[1]
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    print('Will save this model to', train_dir)
    bsize=128
    no_input_channels=4
    block_size=8
    timesteps=200
    latent_dim=16
    num_transitions=7
    epoch_size = 128
    num_epochs = 100
    data_dir = './labeled_data/'
    labelers = ['labit', 'ffelici', 'maurizio']
    labelers=['ffelici']
    shuffle=True
    stride = int(16)
    conv_w_size = 32
    look_ahead = 8
    # lstm_time_spread = int(256)
    source_words_per_sentence = [1,2,4, 8, 16, 32]
    target_words_per_sentence = [1,2,4, 8, 16, 32]
    
    
    # # 
    block_size=10
    stride = int(10)
    conv_w_size = 40
    look_ahead = 10
    source_words_per_sentence = [17]
    target_words_per_sentence = [18]
    
    all_shots = [61057,57103,26386,33459,43454,34010,32716,32191,61021,33638,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,
                31650,31554,42514,26383,48580,62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,42197,
                33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,30262,45103,33446,33567] #39872 42062
    train_shots = all_shots[:len(all_shots)//2]
    val_shots = all_shots[len(all_shots)//2:]
    train_shots = [61057,57103,26386,33459,43454,34010,32716,32191,61021,33638,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,
                31650,31554,42514,26383,48580]
    val_shots = [62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,42197,
                33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,30262,45103,33446,33567]
    
    train_shots = [53601, 47962, 61021, 31839, 33638,
                   31650, 31718, 45103, 32592, 30044,
    33567, 26383, 52302, 32195, 26386, 59825, 33271,
    56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103,]
    
    val_shots = [30268, 61057, 30290, 30197, 43454, 30310,
    60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225,
    29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218,
    32191, 58460, 31554, 30262, 45105,]
    # 
    train_shots = [26383, 26386, 53601, 47962, 61021, 31839, 33638,
                    31650, 31718, ]
                   # 32592, 30044,]
    val_shots = [26383, 26386, 53601, 47962, 61021, 31839, 33638,
                    31650, 31718, ]
                   # 32592, 30044,]
    # 
    print('randomized train shot ids', train_shots)
    print('randomized val shots ids', val_shots)
    params_exp = {
                'batch_size': int(bsize),
                'conv_w_size':conv_w_size,
                'no_input_channels' : no_input_channels,
                'block_size':block_size,
                'lstm_time_spread': int(timesteps),
                'latent_dim': latent_dim,
                'n_classes': num_transitions,
                'epoch_size': epoch_size,
                'num_epochs': num_epochs,
                'stride':int(stride),
                'data_dir':data_dir,
                'labelers':labelers,
                'shuffle':shuffle,
                'source_words_per_sentence': source_words_per_sentence,
                'target_words_per_sentence':target_words_per_sentence,
                'look_ahead':look_ahead,
                'train_data_name':'',
                }
    # print('experiment parameters', params_exp)
    params_random_train = {'shot_ids': train_shots}
    params_random_train.update(params_exp)
    params_random_val = {'shot_ids': val_shots}
    params_random_val.update(params_exp)
    save_dic(params_random_train, train_dir + '/params_data_train')
    save_dic(params_random_val, train_dir + '/params_data_test')
    
    # training_generator = SequenceToSequenceDataGenerator(**params_random_train)
    training_generator = []
    # gen_train = next(iter(training_generator))
    # val_generator = SequenceToSequenceDataGenerator(**params_random_val)
    val_generator = []
    # gen_val = next(iter(val_generator))
    train(train_dir, conv_w_size, no_input_channels, latent_dim, num_transitions, num_epochs, training_generator, val_generator, epoch_size, bsize)
    # predict(data_dir, train_dir, num_transitions, block_size, timesteps, conv_w_size, stride, no_input_channels, val_shots, 0, 20000)
    print('Finished. ')
    
if __name__ == '__main__':
    main(sys.argv)
