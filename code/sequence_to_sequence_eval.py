from sequence_to_sequence_funcs import predict
import sys
from sequence_to_sequence_data_generator import *

def main(args):
    train_dir = './experiments/' + args[1]
    params_exp = load_dic(train_dir + '/params_data_test')
    conv_w_size=params_exp['conv_w_size']
    no_input_channels=params_exp['no_input_channels']
    block_size=params_exp['block_size']
    timesteps=params_exp['lstm_time_spread']
    num_transitions=params_exp['n_classes']
    stride = params_exp['stride']
    data_dir = params_exp['data_dir']
    val_shots = params_exp['shot_ids']
    latent_dim = params_exp['latent_dim']
    max_source_sentence_words = max(params_exp['source_words_per_sentence'])
    max_source_sentence_chars = stride * (max_source_sentence_words-1) + conv_w_size
    look_ahead = params_exp['look_ahead']
    labelers = params_exp['labelers']
    print(params_exp)
    # predict(data_dir, train_dir, labelers, num_transitions, block_size, max_source_sentence_chars, conv_w_size,
    #         stride, latent_dim, no_input_channels, val_shots, max_source_sentence_words, look_ahead, 0, 30000) 
    # 
    params_exp = load_dic(train_dir + '/params_data_train')
    train_shots = params_exp['shot_ids']
    predict(data_dir, train_dir, labelers, num_transitions, block_size, max_source_sentence_chars, conv_w_size,
            stride, latent_dim, no_input_channels, train_shots, max_source_sentence_words, look_ahead, 0, 30000) 
    
    print('Finished. ')
    
if __name__ == '__main__':
    main(sys.argv)