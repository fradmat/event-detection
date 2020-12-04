from sequence_to_sequence_data_generator import *
from helper_funcs import load_fshot_from_labeler, normalize_signals_mean, load_exp_params
from keras import backend as K
import os
import matplotlib as mpl
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tf_seq_to_seq import train#, predict_beam, predict_argmax
mpl.use('Agg')
from operator import itemgetter
import sys

def main(args):
    train_dir = '../experiments/' + sys.argv[1]
    params_fixed = load_exp_params(train_dir)
    print('Will save this model to', train_dir)
    dropout=False
    teacher_forcing = True
    normalize_per_shot = True
    if params_fixed['dropout'] == 'True':
        dropout=True
    if params_fixed['teacher_forcing'] == 'False':
        teacher_forcing = False
    if params_fixed['normalize_per_shot'] == 'False':
        normalize_per_shot = False
    # all_shots = [str(n) for n in params_fixed['all_shots'].replace(' ', '').split(',')]
    num_k_folds = int(params_fixed['num_k_folds'])
    val_shots = params_fixed['val_shots']
    train_shots = params_fixed['train_shots']
    params_train = {
                'batch_size': int(params_fixed['batch_size']),
                'conv_w_size':int(params_fixed['conv_w_size']),
                'no_input_channels' : int(params_fixed['no_input_channels']),
                'block_size':int(params_fixed['block_size']),
                'lstm_time_spread': int(params_fixed['lstm_time_spread']),
                'latent_dim': int(params_fixed['latent_dim']),
                'n_classes': int(params_fixed['n_classes']),
                'epoch_size': int(params_fixed['epoch_size']), #
                'num_epochs': int(params_fixed['num_epochs']), #
                'stride':int(params_fixed['stride']),
                'data_dir': '../data/',
                'labelers':params_fixed['labelers'].replace(' ', '').split(','),
                'shuffle':bool(params_fixed['shuffle']),
                'source_words_per_sentence': [int(n) for n in params_fixed['source_words_per_sentence'].replace(' ', '').split(',')],
                'target_words_per_sentence':[int(n) for n in params_fixed['target_words_per_sentence'].replace(' ', '').split(',')],
                'look_ahead':int(params_fixed['look_ahead']),
                'train_data_name':'',
                'gaussian_hinterval':5,
                'machine_id': params_fixed['machine_id'],
                'normalize_per_shot':normalize_per_shot
                }
    print('---------------------------------------------------------------')
    print('Using data from ' + params_fixed['machine_id'])
    print('---------------------------------------------------------------')
    all_shots = os.listdir('../data')
    shots_per_fold = len(all_shots) // num_k_folds
    np.random.shuffle(all_shots)
    # print(all_shots, len(all_shots))
    # exit(0)
    params_exp_random = dict(params_train)
    params_exp_random['k_folds']= {}
    num_train_folds = np.max([num_k_folds - 1, 1])
    num_train_shots_p_k = num_train_folds * shots_per_fold
    
    decoder_type = str(params_fixed['decoder_type'])
    params_exp_random['decoder_type'] = decoder_type
    # # params_random_val['decoder_type'] = decoder_type
    conv_num_filters = int(params_fixed['conv_num_filters'])
    params_exp_random['conv_num_filters'] = conv_num_filters
    params_exp_random['teacher_forcing'] = teacher_forcing
    # # params_random_val['conv_num_filters'] = conv_num_filters
    conv_dense_size = int(params_fixed['conv_dense_size'])
    params_exp_random['conv_dense_size'] = conv_dense_size
    params_exp_random['dropout'] = dropout
    # print(params_exp_random)
    # exit(0)
    
    for k in range(num_k_folds):
        all_ids = np.arange(len(all_shots))
        indexes = (np.arange(shots_per_fold*k, shots_per_fold*k+ num_train_shots_p_k ) + shots_per_fold) %len(all_shots)
        if num_k_folds == 1:
            indexes = indexes[:int(.8*len(indexes))]
        train_ids = all_ids[indexes]
        val_ids = np.delete(all_ids, indexes)
        # print(train_ids, val_ids)
        train_shots = list(np.take(all_shots, train_ids))
        # train_shots = [53601, 47962, 61021, 31839, 33638, 31650, 31718, 45103, 32592, 30044, 33567, 26383, 52302, 32195, 26386, 59825, 33271, 56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103]
        old_val_shots = val_shots
        val_shots = list(np.take(all_shots, val_ids))
        # val_shots = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105]
        # val_shots = []
        print(train_shots)
        print(val_shots)
        print('intersection', set(train_shots) & set(val_shots))
        print('intersection', set(old_val_shots) & set(val_shots))
        # continue
       
       
        train_shots = set(train_shots)
        val_shots = set(val_shots)
        train_shots = list(train_shots - val_shots)
        val_shots = list(val_shots)
        assert len(set(train_shots) & set(val_shots)) == 0
        print('randomized train shot ids', train_shots, len(train_shots))
        print('randomized val shots ids', val_shots, len(val_shots))
        # continue
        params_train['shot_ids'] = train_shots
        # if k > 0:
        #     # assert len(set(train_shots) & set(params_random_train['shot_ids'])) == len(all_shots ) - 2*shots_per_fold
        #     assert len(set(val_shots) & set(params_exp_random['k_folds'][k]['test'])) == 0 #assert no overlap between k-folds
        # 
        params_exp_random['k_folds'][k+1] = {}
        params_exp_random['k_folds'][k+1]['train'] = train_shots
        # print(type(train  _shots))
        params_exp_random['k_folds'][k+1]['test'] = val_shots
        # print(params_exp_random['k_folds'][k+1]['train'])
        # exit(0)
        print('Training on k-fold', k+1, '...')
        # continue
        training_generator = SequenceToSequenceStateGenerator(**params_train)
        params_exp_random['k_folds'][k+1]['norm_factors'] = training_generator.norm_factors
        
        # val_generator = SequenceToSequenceDataGenerator(**params_random_val)
        # exit(0)
        val_generator = []
        save_dic(params_exp_random, train_dir + '/params_exp')
        encoder, decoder, loss_history = train(k+1,
                                train_dir,
                                params_train['conv_w_size'],
                                params_train['no_input_channels'],
                                params_train['latent_dim'],
                                params_train['n_classes'],
                                params_train['num_epochs'],
                                training_generator,
                                val_generator,
                                params_train['epoch_size'],
                                params_train['batch_size'],
                                decoder_type,
                                conv_num_filters,
                                conv_dense_size,
                                params_exp_random['dropout'],
                                teacher_forcing)
    loss_history = np.asarray(loss_history)
    print('saving loss history, loss shape = ', loss_history.shape)
    # print(loss_history)
    np.save(train_dir + '/loss_history.npy', loss_history)
    print('Finished. ')
    
if __name__ == '__main__':
    print('running')
    # exit(0)
    main(sys.argv)
