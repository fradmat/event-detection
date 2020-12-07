import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import sys
from sequence_to_sequence_data_generator import *
tf.keras.backend.set_floatx('float32')
from tf_seq_to_seq import *
import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use('Agg')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

def eval_main(args):
    train_dir = '../experiments/' + args[0]
    params_exp = load_dic(train_dir + '/params_exp')
    # print('-------------------------------------------------------------Experiment Parameters-------------------------------------------------------------------')
    # for param in params_exp.keys():
    #     print(param, ':', params_exp[param])
    # exit(0)
    beam_width = int(args[1])
    chkpt = args[2]
    train_or_val = args[3]
    # print('beam_width :', beam_width)
    # print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    sys.stdout.flush()
    conv_w_size=params_exp['conv_w_size']
    num_input_channels=params_exp['no_input_channels']
    block_size=params_exp['block_size']
    # timesteps=params_exp['lstm_time_spread']
    num_transitions=params_exp['n_classes']
    stride = params_exp['stride']
    
    
    latent_dim = params_exp['latent_dim']
    max_source_sentence_words = max(params_exp['source_words_per_sentence'])
    max_source_sentence_chars = stride * (max_source_sentence_words-1) + conv_w_size
    look_ahead = params_exp['look_ahead']
    max_target_words = max(params_exp['target_words_per_sentence'])
    labelers = params_exp['labelers']
    # labelers = ['apau_and_marceca',]
    batch_size = 1
    
    decoder_type = params_exp['decoder_type']
    conv_num_filters = params_exp['conv_num_filters']
    conv_dense_size = params_exp['conv_dense_size']
    # print(decoder_type)
    # print(args[0])
    # exit(0)
    # machine_id = params_exp['machine_id']
    machine_id = 'DUMMY_MACHINE'
    data_dir = params_exp['data_dir'] 
    
    k_indexes_train_folds = []
    k_indexes_val_folds = []
    # for k_fold in params_exp['k_folds'].keys():
    decoder_type, decoder_type_spec = decoder_type.split('-')
    # print(decoder_type, decoder_type_spec)
    
    norm_factors = {}
    
    for k_fold in params_exp['k_folds'].keys():
        
        k_fold_vals = params_exp['k_folds'][k_fold]
        train_shots = k_fold_vals['train']
        val_shots = k_fold_vals['test']
        #norm_factors = params_exp['k_folds'][k_fold]['norm_factors']
        norm_factors = {}
        if len(params_exp['k_folds'].keys()) > 1:
            print('-----------------fetching results for k-fold', k_fold, '-------------------')
            print('Current k-fold:', k_fold, '. Train shots:', train_shots,' Validation shots:', val_shots)
        #print(norm_factors)
        #exit(0)
        
        encoder = Encoder(latent_dim, batch_size, conv_w_size, num_input_channels, conv_num_filters, conv_dense_size, dropout=False)
        #print(encoder.layers)
        #exit(0)
        
        if decoder_type == 'luong':
            decoder = DecoderLuong(latent_dim, batch_size, num_transitions, decoder_type_spec)
        elif decoder_type == 'attentionless':
            decoder = AttentionlessDecoder(latent_dim, batch_size, num_transitions)
        else:
            decoder = DecoderBahdanau(latent_dim, batch_size, num_transitions)
        checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder) #optimizer = tf.keras.optimizers.Adam() #optimizer=optimizer, 
        checkpoint_dir = './' + train_dir + '/k_fold_' + str(k_fold) + '/training_checkpoints'
        if os.path.isdir(checkpoint_dir):
            f = tf.train.latest_checkpoint(checkpoint_dir)
            # print('----------------------------------------------------------------------------------------------------')
            f = checkpoint_dir + '/ckpt-' + chkpt
            # print('----------------------------------------------------------------------------------------------------')
            print('fetching checkpoint from file', f)
            print('----------------------------------------------------------------------------------------------------')
            sys.stdout.flush()
            checkpoint.restore(f).expect_partial()
        else:
            print('could not find checkpoint, exiting...')
            exit(0)
        # val_shots = []
        # val_shots_ale_gino = [59073, 61714, 61274, 59065, 61010, 61043, 64770, 64774, 64369, 64060, 64662, 64376, 57093, 57095] #ale_gino
        # # val_shots.extend(val_shots_ale_gino)
        # 
        # val_shots_felici_maurizio_labit = [61021, 32911, 30268, 45105, 62744, 60097, 58460, 61057, 31807, 33459, 34309, 53601, 42197] #felici_maurizio_labit
        # # val_shots.extend(val_shots_felici_maurizio_labit)
        # val_shots_old_paper = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514,
        #            62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105] #old_paper
        # val_shots.extend(val_shots_old_paper)
            
        train_shots = set(train_shots)
        val_shots = set(val_shots)
        train_shots = list(train_shots - val_shots)
        val_shots = list(val_shots)    
        # print('train shot ids', train_shots, len(train_shots))
        # print('val shots ids', val_shots, len(val_shots))
        if train_or_val == 'train':
            shot_ids = train_shots
        elif train_or_val == 'val':
            shot_ids = val_shots
            
        print('will predict on the following', len(shot_ids), train_or_val, 'shots:', sorted(shot_ids))
        # exit(0)
        # simple_mean_k_indexes, weighted_mean_k_indexes = predict_beam(k_fold, data_dir, train_dir, labelers, encoder, decoder,
        #                                                          num_transitions, block_size, max_source_sentence_chars, conv_w_size,
        #                                                         stride, latent_dim, num_input_channels, train_shots,
        #                                                         max_source_sentence_words, max_target_words, look_ahead,
        #                                                         0,30000, 'train', beam_width, decoder_type) #2250 400 30000
        # k_indexes_train_folds.append(simple_mean_k_indexes)

        
        if len(shot_ids) > 0:
            simple_mean_k_indexes, weighted_mean_k_indexes = predict_beam(k_fold, data_dir, train_dir, labelers, machine_id, encoder, decoder,
                                                                     num_transitions, block_size, max_source_sentence_chars, conv_w_size,
                                                                    stride, latent_dim, num_input_channels, shot_ids,
                                                                    max_source_sentence_words, max_target_words, look_ahead,
                                                                    0,1000000000, train_or_val, beam_width, decoder_type, chkpt, norm_factors,verbose=False) #2250 400 30000
            k_indexes_val_folds.append(simple_mean_k_indexes)

    # print(encoder.layers[0])
    # print(encoder.summary())
    # print(decoder.summary())
    # k_indexes_train_folds = np.asarray(k_indexes_train_folds)
    # k_indexes_val_folds = np.asarray(k_indexes_val_folds)
    # print(k_indexes_train_folds.shape)
    # print(np.mean(k_indexes_train_folds, axis=1))
    # print(k_indexes_val_folds.shape)
    # print(np.mean(k_indexes_val_folds, axis=1))
    print('Finished. ')
    
if __name__ == '__main__':
    eval_main(sys.argv[1:])