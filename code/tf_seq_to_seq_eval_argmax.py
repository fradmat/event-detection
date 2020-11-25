import sys
from sequence_to_sequence_data_generator import *
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tf_seq_to_seq import *
import os.path
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

def main(args):
    train_dir = './experiments/' + args[1]
    params_exp = load_dic(train_dir + '/params_exp')
    print('-------------------------------------------------------------Experiment Parameters-------------------------------------------------------------------')
    for param in params_exp.keys():
        print(param, ':', params_exp[param])
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    sys.stdout.flush()
    conv_w_size=params_exp['conv_w_size']
    num_input_channels=params_exp['no_input_channels']
    block_size=params_exp['block_size']
    # timesteps=params_exp['lstm_time_spread']
    num_transitions=params_exp['n_classes']
    stride = params_exp['stride']
    data_dir = params_exp['data_dir']
    
    latent_dim = params_exp['latent_dim']
    max_source_sentence_words = max(params_exp['source_words_per_sentence'])
    max_source_sentence_chars = stride * (max_source_sentence_words-1) + conv_w_size
    look_ahead = params_exp['look_ahead']
    max_target_words = max(params_exp['target_words_per_sentence'])
    labelers = params_exp['labelers']
    batch_size = 1
    chkpt = args[2]
    decoder_type = params_exp['decoder_type']
    conv_num_filters = params_exp['conv_num_filters']
    conv_dense_size = params_exp['conv_dense_size']
    decoder_type = params_exp['decoder_type']
    k_indexes_train_folds = []
    k_indexes_val_folds = []
    # for k_fold in params_exp['k_folds'].keys():
    for k_fold in params_exp['k_folds'].keys():
        print('---------------------------------------------------fetching results for k-fold', k_fold, '-------------------------------------------------------------')
        k_fold_vals = params_exp['k_folds'][k_fold]
        train_shots = k_fold_vals['train']
        val_shots = k_fold_vals['test']
        print('Current k-fold:', k_fold, '. Train shots:', train_shots,' Validation shots:', val_shots)
        
        encoder = Encoder(latent_dim, batch_size, conv_w_size, num_input_channels, conv_num_filters, conv_dense_size, dropout=False)
        if decoder_type == 'luong':
            decoder = DecoderLuong(latent_dim, batch_size, num_transitions, 'general')
        elif decoder_type == 'attentionless':
            decoder = AttentionlessDecoder(latent_dim, batch_size, num_transitions)
        else:
            decoder = DecoderBahdanau(latent_dim, batch_size, num_transitions)
        checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder) #optimizer = tf.keras.optimizers.Adam() #optimizer=optimizer, 
        checkpoint_dir = './' + train_dir + '/k_fold_' + str(k_fold) + '/training_checkpoints'
        if os.path.isdir(checkpoint_dir):
            f = tf.train.latest_checkpoint(checkpoint_dir)
            print('----------------------------------------------------------------------------------------------------')
            f = checkpoint_dir + '/ckpt-' + chkpt
            print('----------------------------------------------------------------------------------------------------')
            print('fetching checkpoint from file', f)
            sys.stdout.flush()
            checkpoint.restore(f).expect_partial()
        else:
            print('could not find checkpoint, exiting...')
            exit(0)
        # exit(0)
        simple_mean_k_indexes, weighted_mean_k_indexes = predict_argmax(k_fold, data_dir, train_dir, labelers, encoder, decoder,
                                                                 num_transitions, block_size, max_source_sentence_chars, conv_w_size,
                                                                stride, latent_dim, num_input_channels, train_shots,
                                                                max_source_sentence_words, max_target_words, look_ahead,
                                                                0,30000, 'train', decoder_type) #2250 400 30000
        k_indexes_train_folds.append(simple_mean_k_indexes)
        if len(val_shots) > 0:
            simple_mean_k_indexes, weighted_mean_k_indexes = predict_argmax(k_fold, data_dir, train_dir, labelers, encoder, decoder,
                                                                     num_transitions, block_size, max_source_sentence_chars, conv_w_size,
                                                                    stride, latent_dim, num_input_channels, val_shots,
                                                                    max_source_sentence_words, max_target_words, look_ahead,
                                                                    0,30000, 'test', decoder_type) #2250 400 30000
            k_indexes_val_folds.append(simple_mean_k_indexes)

    print('Finished. ')
    
if __name__ == '__main__':
    main(sys.argv)