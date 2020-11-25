from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPool1D, TimeDistributed, LSTM, Layer, InputLayer, Bidirectional, Dropout
from tensorflow.keras import Model, Input
# from tensorflow.keras.metrics import CategoricalAccuracy
from helper_funcs import load_fshot_from_number, normalize_signals_mean, det_trans_to_state, get_trans_ids, k_statistic
from sequence_to_sequence_data_generator import *
from plot_shot_results import plot_shot_full_seq2seq, plot_shot_simplified, plot_attention_matrix, plot_attention_prediction, plot_conf_mat
from plot_scores import plot_kappa_histogram, out_sorted_scores
from operator import attrgetter

import sys
import numpy as np
import os
import io
import time
import pickle
from datetime import datetime
from pathlib import Path


# import warnings


class ConvolutionalFilters(Layer):
    def __init__(self,num_filters, dense_size, dropout): # conv_w_size, num_channels
        super(ConvolutionalFilters, self).__init__()
        self.num_filters = num_filters
        self.dense_size = dense_size
        self.dropout = dropout
        # print(self.num_filters, self.dense_size, self.dropout)
        # exit(0)
        self.conv_1 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        self.conv_2 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        self.pool_1 = MaxPool1D(2)
        self.dropout_1 = Dropout(0.5)
        
        self.conv_3 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        self.conv_4 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        self.conv_5 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        self.pool_2 = MaxPool1D(2)
        self.dropout_2 = Dropout(0.5)
        
        # self.conv_6 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        # self.conv_7 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        # self.conv_8 = Conv1D(self.num_filters, 3, activation='relu', padding='same')
        # self.pool_3 = MaxPool1D(2)
        
        self.flatten = Flatten()
        self.dense_1 = Dense(self.dense_size, activation = 'relu')
        self.dense_2 = Dense(self.dense_size, activation = 'relu')
        self.dropout_3 = Dropout(0.5)
      
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        if self.dropout:
            x = self.dropout_1(x)
        x = self.pool_1(x)
        
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        if self.dropout:
            x = self.dropout_2(x)
        x = self.pool_2(x)
        # 
        # x = self.conv_6(x)
        # x = self.conv_7(x)
        # x = self.conv_8(x)
        # x = self.pool_3(x)
        # 
        x = self.flatten(x)
        x = self.dense_1(x)
        outputs = self.dense_2(x)
        if self.dropout:
            outputs = self.dropout_3(outputs)
        return outputs

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, batch_sz, conv_w_size, num_channels, conv_num_filters, conv_dense_size, dropout):
    # def __init__(self, vocab_size, embedding_dim, latent_dim, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.latent_dim = latent_dim
        self.conv_filters = TimeDistributed(ConvolutionalFilters(conv_num_filters, conv_dense_size, dropout))
        self.lstm = LSTM(self.latent_dim,return_sequences=True,return_state=True)
  
    def call(self, inputs, verbose=False): #hidden
        if verbose:
            start = datetime.now()
            print('computing convs.,',  (datetime.now() - start).microseconds/1000)
        conv_time_dist = self.conv_filters(inputs)
        if verbose:
            print('computed convs. computing lstm.,',  (datetime.now() - start).microseconds/1000)
        output, h, c = self.lstm(conv_time_dist)
        if verbose:
            print('computed lstm.,',  (datetime.now() - start).microseconds/1000)
        return output, [h, c]
  
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(latent_dim)
        self.W2 = tf.keras.layers.Dense(latent_dim)
        self.V = tf.keras.layers.Dense(1)
  
    def call(self, prev_hidden, encoder_output):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, 19, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        # print(prev_hidden[0].shape, prev_hidden[1].shape)
        # print(tf.concat(prev_hidden, axis=-1))
        prev_hidden_with_t_axis = tf.expand_dims(tf.concat(prev_hidden, axis=-1), 1)
    
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, 19, units)
        score = self.V(tf.nn.tanh(self.W1(prev_hidden_with_t_axis) + self.W2(encoder_output)))
        # print(score.shape)
        attention_weights = tf.nn.softmax(score, axis=1)
        # print(attention_weights.shape)
        # exit(0)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1) #weighted average
        # print(encoder_output.shape, prev_hidden_with_t_axis.shape, score.shape, attention_weights.shape, context_vector)
        # exit(0)
        shape = attention_weights.shape
        return context_vector, tf.reshape(attention_weights, (shape[0], shape[2], shape[1]))

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, latent_dim, method='general'):
        super(LuongAttention, self).__init__()
        self.method = method  
        if self.method == 'concat':
            self.W1 = tf.keras.layers.Dense(latent_dim * 2) #latent_dim * 2 because we will concatenate the encoder output (lat_dim) with a repeated (lat_dim) decoder lstm output
            # self.W2 = tf.keras.layers.Dense(latent_dim)
            self.V = tf.keras.layers.Dense(1, activation = 'tanh')
        elif self.method == 'general':
            self.W = tf.keras.layers.Dense(latent_dim)
  
    def call(self, decoder_lstm_output, encoder_output):
        # print(decoder_lstm_output.shape, encoder_output.shape, self.W(decoder_lstm_output).shape, self.W(encoder_output).shape)
        # print(decoder_lstm_output.shape, tf.repeat(decoder_lstm_output, 27, axis=1).shape)
        if self.method == 'general':
            score = tf.matmul(decoder_lstm_output, self.W(encoder_output), transpose_b = True) # score will have shape: (batch_size, 1, max_len)
        elif self.method == 'dot':
            score = tf.matmul(decoder_lstm_output, encoder_output, transpose_b = True)
        #concat might not be correctly implemented, will have to re-check later
        elif self.method == 'concat':
            decoder_lstm_output = tf.repeat(decoder_lstm_output, encoder_output.shape[1], axis=1)
            # print(self.W1(tf.concat([decoder_lstm_output, encoder_output], axis=-1)).shape)
            score = self.V(self.W1(tf.concat([decoder_lstm_output, encoder_output], axis=-1)))
            score = tf.transpose(score, [0, 2, 1]) 
        # print(score.shape)
        alignment_vector = tf.nn.softmax(score, axis=-1) #apply softmax at each timestep (which is just 1)
        # print(alignment_vector)
        # exit(0)
        context_vector = tf.matmul(alignment_vector, encoder_output)
        # print(context_vector.shape)
        # exit(0)
        return context_vector, alignment_vector  
  
class Decoder(Model):
    def __init__(self, latent_dim, batch_sz, num_transitions):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.latent_dim = latent_dim #2* because of bi-directional lstm in encoder
        self.num_transitions = num_transitions
        self.lstm = LSTM(self.latent_dim,return_sequences=True,return_state=True)
    
    def initialize_hidden_states(self):
        return [tf.zeros((self.batch_sz, self.latent_dim), dtype=tf.dtypes.float32), tf.zeros((self.batch_sz, self.latent_dim), dtype=tf.dtypes.float32)] #h, c. self.latent dim is already 2* global latent dim
    
    def initialize_context(self):
        return tf.zeros((self.batch_sz, self.latent_dim), dtype=tf.dtypes.float32) 
  
class DecoderBahdanau(Decoder):
    def __init__(self, latent_dim, batch_sz, num_classes):
        super(DecoderBahdanau, self).__init__(latent_dim, batch_sz, num_classes)
        # self.t_dist_1 = TimeDistributed(Dense(64, activation = 'relu'))
        self.fc = tf.keras.layers.Dense(num_classes, activation = 'softmax')
        self.bahdanau = BahdanauAttention(self.latent_dim)
        
    def call(self, dec_input, states, encoder_output, lstm_out_prev):
        context_vector, attention_weights = self.bahdanau(states, encoder_output)
        # x shape after concatenation == (batch_size, 1, num_trans + hidden_size)
        lstm_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        # passing the concatenated vector to the LSTM
        lstm_output, h, c = self.lstm(lstm_input, initial_state = states)
        lstm_output = tf.reshape(lstm_output, (-1, lstm_output.shape[2])) #squash time dimension
        x = self.fc(lstm_output)
        return x, [h, c], attention_weights, lstm_output

class DecoderLuong(Decoder):
    def __init__(self, latent_dim, batch_sz, num_classes, attention_type='general'):
        super(DecoderLuong, self).__init__(latent_dim, batch_sz, num_classes)
        self.wc = tf.keras.layers.Dense(latent_dim, activation='tanh', name='wc')
        self.ws = tf.keras.layers.Dense(num_classes, name='ws', activation='softmax')
        # self.w_concat = tf.keras.layers.Dense(1, name='w_concat')
        self.luong = LuongAttention(self.latent_dim, attention_type)
  
    def call(self, dec_input, states, encoder_output, lstm_out_prev, verbose=False):
        if verbose:
            start = datetime.now()
            print('calling decoder')
        # uncomment to lstm_out_prev to next time step
        lstm_input = dec_input
        # print(lstm_input.shape, lstm_out_prev.shape)
        lstm_input = tf.concat([lstm_input, tf.expand_dims(lstm_out_prev, 1)], axis=2)
        
        lstm_output, h, c = self.lstm(lstm_input, initial_state = states)
        if verbose:
            print('computed lstm step,', (datetime.now() - start).microseconds/1000)
        context, alignment = self.luong(lstm_output, encoder_output)
        if verbose:
            print('computed attention,', (datetime.now() - start).microseconds/1000)
        lstm_output = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_output, 1)], axis = 1) #paper equation 5. squeeze to remove time dimension
        lstm_output = self.wc(lstm_output) #paper equation 5
        x = self.ws(lstm_output)  #paper equation 6
        return x, [h,c], alignment, lstm_output

class AttentionlessDecoder(Decoder):
    def __init__(self, latent_dim, batch_sz, num_classes):
        super(AttentionlessDecoder, self).__init__(latent_dim, batch_sz, num_classes)
        self.dense = tf.keras.layers.Dense(32, activation = 'relu')
        self.fc = tf.keras.layers.Dense(num_classes, activation = 'softmax')
      
    def call(self, dec_input, states, verbose=False):
        if verbose:
            start = datetime.now()
            print('calling decoder')
        lstm_output, h, c = self.lstm(dec_input, initial_state = states)
        if verbose:
            print('computed lstm step,', (datetime.now() - start).microseconds/1000)
        x = self.dense(lstm_output)
        x = self.fc(x)
        # print(x.shape, lstm_output.shape)
        # exit(0)
        if verbose:
            print('computed dense,', (datetime.now() - start).microseconds/1000)
        return tf.squeeze(x, 1), [h, c]

def train_step(inputs, targets, encoder, decoder, optimizer, loss_object, metric_object, teacher_forcing, decoder_type):
    # print('in train step')
    loss = 0
    acc = 0
    # exit(0)
    encoder_inputs = inputs['encoder_inputs']
    decoder_inputs = inputs['decoder_inputs']
    decoder_outputs = targets['decoder_outputs']
    # print('data shape at this step:', encoder_inputs.shape)
    with tf.GradientTape() as tape:
        # print(encoder_inputs.shape)
        # exit(0)
        encoder_output, encoder_states = encoder(encoder_inputs)
        # print(encoder_output.shape)
        # exit(0)
        decoder_hidden = encoder_states
        # decoder_hidden = decoder.initialize_hidden_states()
        # decoder.reset_states()
        lstm_output = decoder.initialize_context()
        # Teacher forcing - feeding the target as the next input
        predictions = decoder_inputs[:,0,:] #without teacher forcing
        for t in range(decoder_inputs.shape[1]): #for each time slice
            # passing enc_output to the decoder
            if teacher_forcing == True:
                decoder_input_t = decoder_inputs[:,t:t+1,:] #with teacher forcing #(batch_sample, timestep, num_classes)
            else: #without teacher forcing
                arg_maxs = tf.math.argmax(predictions, axis=-1)
                to_cat = tf.keras.utils.to_categorical(arg_maxs, decoder_inputs.shape[2])
                decoder_input_t = (tf.expand_dims(to_cat, 1))
                # decoder_input_t = tf.expand_dims(predictions, 1) 
            decoder_output_t = decoder_outputs[:,t,:] #:t+1            
            if decoder_type == 'attentionless':
                predictions, decoder_hidden, = decoder(decoder_input_t, decoder_hidden)
            else:
                predictions, decoder_hidden, _, lstm_output = decoder(decoder_input_t, decoder_hidden, encoder_output, lstm_output)
            # print(predictions.shape)
            # print(decoder_output_t.shape)
            # exit(0)
            assert decoder_output_t.shape == predictions.shape
            loss += loss_object(decoder_output_t, predictions)
            acc += metric_object(decoder_output_t, predictions)
        
      # exit(0)
    batch_loss = (loss / int(decoder_outputs.shape[1]))
    batch_acc = (acc / int(decoder_outputs.shape[1]))
  
    variables = encoder.trainable_variables + decoder.trainable_variables
  
    gradients = tape.gradient(loss, variables)
  
    optimizer.apply_gradients(zip(gradients, variables))
  
    return batch_loss, batch_acc

def train(k_fold, train_dir, conv_w_size, num_input_channels, latent_dim, num_transitions,
          num_epochs, train_generator, val_generator, epoch_size, bsize, decoder_type, conv_num_filters, conv_dense_size, dropout, teacher_forcing):
  
    BATCH_SIZE = bsize
    steps_per_epoch = epoch_size
    latent_dim = latent_dim
    encoder = Encoder(latent_dim, BATCH_SIZE, conv_w_size, num_input_channels, conv_num_filters, conv_dense_size, dropout)
    decoder_type, decoder_type_spec = decoder_type.split('-')
    # print(decoder_type, decoder_type_spec)
    # exit(0)
    if decoder_type == 'luong':
        decoder = DecoderLuong(latent_dim, BATCH_SIZE, num_transitions, decoder_type_spec) #decoder_type_spec = general, concat, ...
    elif decoder_type == 'attentionless':
        decoder = AttentionlessDecoder(latent_dim, BATCH_SIZE, num_transitions)
    else:
        decoder = DecoderBahdanau(latent_dim, BATCH_SIZE, num_transitions)
    EPOCHS = num_epochs
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    metric_object = tf.keras.metrics.CategoricalAccuracy()
    
    saves_dir = './' + train_dir + '/k_fold_' + str(k_fold)
    # if not os.path.isdir(saves_dir):
    #   print('could not find directory for checkpoint and logs, exiting...')
    #   exit(0)
    checkpoint_dir = saves_dir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder) #optimizer=optimizer, 
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = saves_dir + '/logs/' + current_time
    # test_log_dir = saves_dir + 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    print('----------------------------------------------------------')
    print('Starting training...')
    loss_history =[]
    for epoch in range(EPOCHS):
        start = time.time()
      
        metric_object.reset_states()
      # loss_object.reset_states()
        total_loss = 0
        total_acc = 0
        for batch_ind, batch in enumerate(next(iter(train_generator))):
            inp = batch[0]
            targ = batch[1]
            # print('going into')
            batch_loss, batch_acc = train_step(inp, targ, encoder, decoder, optimizer, loss_object, metric_object, teacher_forcing, decoder_type)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', batch_loss, step=epoch)
                tf.summary.scalar('accuracy', batch_acc, step=epoch)
      
            total_loss += batch_loss
            total_acc += batch_acc
            print('Epoch {} Batch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1,
                                                         batch_ind + 1,
                                                         batch_loss.numpy(),
                                                         batch_acc.numpy()))
            
            sys.stdout.flush()
            if batch_ind + 1 == epoch_size:
                break
          
      # if epoch + 1 % 10 == 0:
  
        epoch_loss = total_loss / steps_per_epoch
        epoch_acc = total_acc/steps_per_epoch
        print('Epoch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
        loss_history.append(epoch_loss)
        if (epoch+1) % 10 == 0:
            print('saving checkpoint...')
            checkpoint.save(file_prefix = checkpoint_prefix)
    print(encoder.summary())
    print(decoder.summary())
    return encoder, decoder, loss_history

def predict_argmax(k_fold, data_dir, train_dir, labelers, encoder, decoder, num_transitions, block_size, max_source_sentence_chars, conv_w_size, stride, latent_dim,
            num_input_channels, val_shots, max_source_sentence_words, max_train_target_words, look_ahead, pred_start, pred_interval, data_type, decoder_type):
    k_indexes = []
    ground_truth_concat = []
    shot_det_concat = []
    for shot in val_shots:
        print('---------------------------Predicting/evaluating shot', shot, '--------------------------------------------')
        shot_df, fshot_times = load_fshot_from_number(shot, data_dir)
        # start = 2250   #LH transition @ 2450
        shot_df = normalize_signals_mean(shot_df)[pred_start:pred_start+pred_interval]
        shot_signals = get_raw_signals_in_window(shot_df).swapaxes(0,1).astype(np.float32)
        # print(shot_signals.shape)
        # exit(0)
        decoded_sequence, timewise_preds, timewise_preds_cat, attention_weights_sequence= decode_context(
            np.expand_dims(shot_signals, axis=0)[pred_start:pred_start+pred_interval],
            encoder, decoder, num_transitions, block_size,
            max_source_sentence_chars, conv_w_size, stride,
            num_input_channels, max_source_sentence_words, max_train_target_words, look_ahead, decoder_type)
        
        # timewise_preds_cat = np.repeat(timewise_preds_cat, block_size, axis=0)
        # remainder = len(shot_df) - len(timewise_preds_cat) -look_ahead
        # timewise_preds_cat = np.concatenate([np.zeros((look_ahead + block_size, num_transitions)), timewise_preds_cat])
        # timewise_preds_cat = np.concatenate([timewise_preds_cat, np.zeros((remainder - block_size, num_transitions))])
        # # if look-ahead has been used, we assume that there are no transitions in the look-ahead regions
        # # at the start and end of the shot. The same goes for the few remaining time points for which a block does not exist
        # # due to mis-alignment.
        # # furthermore, the repeat of the block values to align with the shot sequence
        # # is done such that the repetition is shifted by block_size.
        # 
        decoded_sequence = np.repeat(decoded_sequence, block_size, axis=0)
        remainder = len(shot_df) - len(decoded_sequence) -look_ahead
        # print(decoded_sequence.shape, np.zeros((look_ahead + block_size, num_transitions)).shape)
        decoded_sequence = np.concatenate([np.ones((look_ahead + block_size, 1))*decoded_sequence[0], decoded_sequence]) #pad initial value
        decoded_sequence = np.concatenate([decoded_sequence, np.ones((remainder - block_size, 1))*decoded_sequence[-1]]) #pad last value
    
        # for t, t_id in enumerate(trans_ids):
        #   shot_df[t_id + '_det'] = timewise_preds_cat[:, t]
        # shot_lhd_det = det_trans_to_state(shot_df)
        # shot_df['LHD_det'] = shot_lhd_det
        # print(decoded_sequence.shape, shot_df.shape)
        shot_df['LHD_det'] = decoded_sequence + 1
        # exit(0)
        shot_signals = get_raw_signals_in_window(shot_df).swapaxes(0,1).astype(np.float32)
        times = shot_df.time.values
        # trans_detected = np.where(decoded_sequence[:, 0]!=6)[0]
        # print('trans_detected', trans_detected)
        path = train_dir+'/k_fold_' + str(k_fold) +'/'#+ data_type + '/'
        fname = path+'/'+str(shot)+'attention_mat_predictions.pdf'
        
        
        
        
        # transitions = np.expand_dims(max_beam.transitions, 1)
        # trans_detected = np.where(transitions[:, 0]!=6)[0]
        # plot_attention_matrix(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot,
        #                           stride, conv_w_size, block_size, fname, max_source_sentence_chars, max_train_target_words,
        #                           look_ahead, num_input_channels)
        # exit(0)
        # fname = train_dir+'/'+str(shot)+'attention_predictions.pdf'
        # plot_attention_prediction needs corrections because of look-ahead and such
        # plot_attention_prediction(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot,
        #                           stride, conv_w_size, block_size, fname, max_source_sentence_chars, num_input_channels)
        
        k_st, ground_truth, ground_truth_cut, shot_det_cut = evaluate(shot, data_dir, train_dir, labelers, shot_df)
        ground_truth_concat.extend(ground_truth_cut)
        shot_det_concat.extend(shot_det_cut)
        k_indexes += [k_st]
        
        shot_df['ELM_prob'] = np.zeros(len(shot_df))
        shot_df['LHD_label'] = ground_truth
        # plot_shot_simplified(shot, shot_df, path+'/'+str(shot)+'prediction.pdf')
        
        # np.save(path +str(shot)+'decoded.npy', decoded_sequence)
        # np.save(path +str(shot)+'timewise.npy', timewise_preds)
        # np.save(path +str(shot)+'states.npy',states_pred)
      
    ground_truth_concat = np.asarray(ground_truth_concat)
    shot_det_concat = np.asarray(shot_det_concat)
    assert ground_truth_concat.shape[0] == shot_det_concat.shape[0]
    weighted_mean_k_indexes = k_statistic(shot_det_concat, ground_truth_concat)
    
    k_indexes = np.asarray(k_indexes)
    print(k_indexes.shape)
    weighted_mean_k_indexes = np.asarray(weighted_mean_k_indexes)
    # print(weighted_k_indexes.shape)
    histo_fname = path + data_type + '_k_ind_histogram.pdf'
    title = ''
    plot_kappa_histogram(k_indexes, histo_fname, title)
    # exit(0)
    simple_mean_k_indexes = np.mean(k_indexes, axis=0)
    print('simple_mean_k_indexes', simple_mean_k_indexes, '. weighted_mean_k_indexes',weighted_mean_k_indexes)

    return simple_mean_k_indexes, weighted_mean_k_indexes

def predict_beam(k_fold, data_dir, train_dir, labelers, machine_id, encoder, decoder, num_transitions, block_size, max_source_sentence_chars,
                 conv_w_size, stride, latent_dim,num_input_channels, val_shots, max_source_sentence_words, max_train_target_words,
                 look_ahead, pred_start, pred_interval, data_type, beam_width, decoder_type, chkpt, norm_factors,verbose=True):
    k_indexes = []
    k_indexes_blocks = []
    ground_truth_concat = []
    ground_truth_block_concat = []
    shot_det_concat = []
    shot_det_block_concat = []
    k_indexes_blocks_per_shot = {}
    path_plots = train_dir+'/k_fold_' + str(k_fold) +'/eval_ckpt_' + chkpt + '/plots/' + data_type + '/' 
    path_values = train_dir+'/k_fold_' + str(k_fold) +'/eval_ckpt_' + chkpt + '/shots/' + data_type  + '/'
    path_logs = train_dir+'/k_fold_' + str(k_fold) + '/logs/' 
    # fpath = path + '/' + chkpt + '/'
    # print(path_plots, path_values)
    # print(os.path.exists(path_plots))
    # if not os.path.exists(path_plots):
    #     os.mkdir(path_plots)
    # if not os.path.exists(path_values):
    #     os.mkdir(path_values)
    
    Path(path_plots).mkdir(parents=True, exist_ok=True)
    Path(path_values).mkdir(parents=True, exist_ok=True)
    # exit(0)
    num_classes=3
    conf_matrix_all = np.zeros((num_classes,num_classes))
    # warnings.filterwarnings("ignore")
    for shot in val_shots:
        print('---------------------------Predicting/evaluating shot', shot, '--------------------------------------------')
        shot_df, fshot_times = load_fshot_from_number(shot, machine_id, data_dir, labelers)
        # start = 2250   #LH transition @ 2450
        # print(shot_df)
        # import matplotlib
        # matplotlib.use('QT4Agg')
        # print(pred_start, pred_interval)
        if norm_factors != {}:
            # print(norm_factors)
            # exit(0)
            print('normalizing shot with norm. factor from train data')
            shot_df = normalize_signals_with_factors(shot_df, norm_factors).reset_index()[pred_start:pred_start+pred_interval]
        else:
            shot_df = normalize_signals_mean(shot_df).reset_index()[pred_start:pred_start+pred_interval]
        # plt.plot(shot_df.IP.values)
        # plt.plot(shot_df.PD.values)
        # plt.plot(shot_df.DML.values)
        # plt.plot(shot_df.FIR.values)
        # plt.show()
        # plt.plot(shot_df.LHD_label.values)
        # plt.show()
        shot_signals = get_raw_signals_in_window(shot_df).swapaxes(0,1).astype(np.float32)
        
        
        
        # exit(0)
        # exit(0)
        
        max_beams = beam_search(
            np.expand_dims(shot_signals, axis=0)[pred_start:pred_start+pred_interval],
            encoder, decoder, num_transitions, block_size,
            max_source_sentence_chars, conv_w_size, stride,
            num_input_channels, max_source_sentence_words, max_train_target_words, look_ahead, beam_width,decoder_type,path_logs,verbose)
        # exit(0)
        max_beam = max_beams[0]
        transitions = np.expand_dims(max_beam.transitions, 1)
        states_blocks = max_beam.plasma_states
        trans_blocks = max_beam.transitions
        attention_weights_sequence = max_beam.attention_weights.swapaxes(1,2)
        
        # timewise_preds_cat = np.repeat(timewise_preds_cat, block_size, axis=0)
        # # print(timewise_preds_cat.shape, shot_df.shape)
        # remainder = len(shot_df) - len(timewise_preds_cat) -look_ahead
        # timewise_preds_cat = np.concatenate([np.zeros((look_ahead + block_size, num_transitions)), timewise_preds_cat])
        # timewise_preds_cat = np.concatenate([timewise_preds_cat, np.zeros((remainder - block_size, num_transitions))])
        # # if look-ahead has been used, we assume that there are no transitions in the look-ahead regions
        # # at the start and end of the shot. The same goes for the few remaining time points for which a block does not exist
        # # due to mis-alignment.
        # # furthermore, the repeat of the block values to align with the shot sequence
        # # is done such that the repetition is shifted by block_size.
        
        
        states_pred = np.repeat(states_blocks, block_size, axis=0)
        remainder = len(shot_df) - len(states_pred) - look_ahead
        # states_pred = np.concatenate([np.repeat(states_pred[0], look_ahead + block_size), states_pred])
        # states_pred = np.concatenate([states_pred, np.repeat(states_pred[-1], remainder - block_size)])
        
        states_pred = np.concatenate([np.repeat(states_pred[0], look_ahead), states_pred])
        states_pred = np.concatenate([states_pred, np.repeat(states_pred[-1], remainder)])
        
        shot_df['LHD_det'] = states_pred
        
        shot_signals = get_raw_signals_in_window(shot_df).swapaxes(0,1).astype(np.float32)
        times = shot_df.time.values
        trans_detected = np.where(transitions[:, 0]!=6)[0]
        
        fname = path_plots+'/'+str(shot)+'_attention_mat_predictions.pdf'
        
        # plot_attention_matrix(shot_signals, states_blocks, shot_df.LHD_label.values, trans_detected, attention_weights_sequence, times, shot,
        #                           stride, conv_w_size, block_size, fname, max_source_sentence_chars, max_train_target_words,
        #                           look_ahead, num_input_channels)
        
        fname = path_plots+'/'+str(shot)+'_attention_predictions.pdf'
        # # 
        # plot_attention_prediction(shot_signals, states_blocks, trans_detected, attention_weights_sequence, times, shot,
        #                           stride, conv_w_size, block_size, fname, max_source_sentence_chars, max_train_target_words,
        #                           look_ahead, num_input_channels)
        
        k_st, ground_truth, ground_truth_cut, shot_det, shot_det_cut, k_st_blocks, ground_truth_blocks, states_blocks, conf = evaluate(shot, data_dir, train_dir, machine_id, labelers, shot_df, states_blocks, block_size, look_ahead)
        
        conf_matrix, mat_order = conf[0], conf[1]
        fname = path_plots+'/'+str(shot)+'_confusion_matrix.pdf'
        # print(conf_matrix)
        plot_conf_mat(conf_matrix, mat_order, fname)
        conf_matrix_all += conf_matrix
        # exit(0)
        ground_truth_concat.extend(ground_truth_cut)
        shot_det_concat.extend(shot_det_cut.LHD_det.values)
        ground_truth_block_concat.extend(ground_truth_blocks)
        shot_det_block_concat.extend(states_blocks)
        k_indexes += [k_st]
        k_indexes_blocks += [k_st_blocks]
        k_indexes_blocks_per_shot[shot] = k_st_blocks
        
        shot_df = (shot_df.loc[shot_df['time'].round(5).isin(shot_det.time.round(5))]).copy()
        
        shot_df['ELM_prob'] = np.zeros(len(shot_df))
        shot_df['LHD_label'] = ground_truth
        plot_shot_simplified(shot, shot_df, path_plots+'/'+str(shot)+'_prediction.pdf')
        # exit(0)
        fname=path_plots+'/'+str(shot)+'_full_prediction.pdf'
        # # plot_shot_full_seq2seq(shot_signals, states_blocks, trans_detected, attention_weights_sequence, times, shot,
        # #                          stride, conv_w_size, block_size, fname, max_source_sentence_chars, max_train_target_words,
        # #                           look_ahead, num_input_channels, shot_df.LHD_label.values)
        # # shot_df['LHD_det'] = np.ones(len(shot_df))
        # # states_blocks = np.ones(len(shot_df)//block_size)[: -10]
    
    
        # plot_shot_full_seq2seq(shot, shot_df, states_blocks, block_size, look_ahead, k_st_blocks, fname)  #,  [0,0,0,0,]

        sys.stdout.flush()
        
        shot_df.to_csv(path_values + 'TCV_' + str(shot) + '_seq2seq_det.csv')
      
        
    print(conf_matrix_all)
    fname = path_plots+'/global_confusion_matrix.pdf'
    plot_conf_mat(conf_matrix_all, mat_order, fname)
    # exit(0)
    ground_truth_concat = np.asarray(ground_truth_concat)
    shot_det_concat = np.asarray(shot_det_concat)
    assert ground_truth_concat.shape[0] == shot_det_concat.shape[0]
    weighted_mean_k_indexes = k_statistic(shot_det_concat, ground_truth_concat)
    
    shot_det_block_concat = np.asarray(shot_det_block_concat)
    ground_truth_block_concat = np.asarray(ground_truth_block_concat)
    weighted_mean_blocks_k_indexes = k_statistic(shot_det_block_concat, ground_truth_block_concat)
    
    k_indexes = np.asarray(k_indexes)
    print(k_indexes.shape)
    weighted_mean_k_indexes = np.asarray(weighted_mean_k_indexes)
    # print(weighted_k_indexes.shape)
    histo_fname = path_plots + data_type + '_k_ind_histogram.pdf'
    title = ''
    # plot_kappa_histogram(k_indexes, histo_fname, title)
    # exit(0)
    simple_mean_k_indexes = np.mean(k_indexes, axis=0)
    print('simple_mean_k_indexes', simple_mean_k_indexes, '. weighted_mean_k_indexes',weighted_mean_k_indexes)
    
    k_indexes_blocks = np.asarray(k_indexes_blocks)
    simple_mean_k_indexes_block = np.mean(k_indexes_blocks, axis=0)
    plot_kappa_histogram(k_indexes_blocks, histo_fname, title)
    print('simple_mean_k_indexes_block', simple_mean_k_indexes_block, '. weighted_mean_k_indexes_block',weighted_mean_blocks_k_indexes)
    
    # print(k_indexes_blocks_per_shot)
    # exit(0)
    
    
    fpath = path_plots + data_type + '_sorted_scores_'
    # print(fpath)
    # exit(0)
    out_sorted_scores(k_indexes_blocks_per_shot, fpath)

    return simple_mean_k_indexes, weighted_mean_k_indexes

def evaluate(shot_id, data_dir, train_dir, machine_id, labelers, shot_predicted, states_blocks, block_size, look_ahead):
    # print('Evaluating shot', shot_id)
    intersect_times = np.round(shot_predicted.time.values,5)
    labeler_states = []
    # print(intersect_times.shape)_existing = []
    labelers_existing = []
    for labeler in labelers:
        shot_id_lab = str(shot_id) + '-' + labeler
        try:
            fshot_labeled, fshot_times = load_fshot_from_labeler(shot_id_lab, machine_id, data_dir)
            intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
            labelers_existing.append(labeler)
            print('Found shot file for shot ' + str(shot_id) + ' from labeler ' + str(labeler))
        # print('intersect_times', len(intersect_times))
        except:
            print('Could not find shot file for shot ' + str(shot_id) + ' from labeler ' + str(labeler))
            continue
        # print(fshot_labeled.shape, intersect_times.shape)
      # exit(0)
    for labeler in labelers_existing:
        shot_id_lab = str(shot_id) + '-' + labeler
        fshot_labeled, fshot_times = load_fshot_from_labeler(shot_id_lab, machine_id, data_dir)
        fshot_equalized = (fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]).copy()
        labeler_states += [fshot_equalized['LHD_label'].values]
    # make sure labels from all labelers have the same length! 
    # print(len(intersect_times))
    # print(fshot_equalized.columns)
    # print(shot_predicted.columns)
    labeler_states = np.asarray(labeler_states)
    print('computing result on ground truth agreed by labelers' + str(labelers_existing))
    # print(labeler_states.shape)
    # exit(0)
  #   
    ground_truth = calc_mode(labeler_states.swapaxes(0,1))
    # print(ground_truth.shape, shot_predicted.shape, (ground_truth!=-1).shape)
    # ground_truth_mask = np.where(ground_truth!=-1)[0].astype(bool)
    # ground_truth_concat = ground_truth[ground_truth_mask]
    shot_predicted = (shot_predicted.loc[shot_predicted['time'].round(5).isin(intersect_times)]).copy()
    # print(ground_truth.shape, shot_predicted.shape)
    assert ground_truth.shape[0] == shot_predicted.shape[0]
    # print(ground_truth.shape)
    # exit(0)
    
    remainder_block = ground_truth.shape[0] % block_size
    quotient_block = ground_truth.shape[0] // block_size
    # print(remainder_block)
    ground_truth_blocks = ground_truth
    if remainder_block != 0:
      ground_truth_blocks = ground_truth[:-remainder_block]
    # print(ground_truth_blocks.shape)
    #
    ground_truth_blocks = ground_truth_blocks.reshape(quotient_block, block_size)
    # print(ground_truth_blocks.shape)
    
    # counts = np.unique(ground_truth_blocks, return_counts = True, axis = 0)
    # print(counts.shape)
    # print(counts[0])
    # 
    # exit(0)
    label_blocks = []
    for block in ground_truth_blocks:
        unique, counts = np.unique(block, return_counts = True)
        counts_dic = dict(zip(unique, counts))
        max_label = max(counts_dic, key=counts_dic.get)
        label_blocks.append(max_label)
    look_ahead_blocks = look_ahead // block_size
    label_blocks = np.asarray(label_blocks)[look_ahead_blocks:] #remove initial look-ahead blocks
    label_blocks = label_blocks[:states_blocks.shape[0]] #remove final look-ahead blocks
    # print(label_blocks.shape, states_blocks.shape)
    # exit(0)
    
    # remove sections of shot where there was no consensus between labelers
    shot_predicted_cut = shot_predicted[(ground_truth!=-1)].copy()
    ground_truth_cut = ground_truth[(ground_truth!=-1)]
    
    gt_mask = (label_blocks != -1)
    ground_truth_blocks = label_blocks[gt_mask]
    states_blocks = states_blocks[gt_mask]
    print(ground_truth_blocks.shape, states_blocks.shape)
    # exit(0)
    
    
    
    # print('calculating with majority and consensual opinion (ground truth)') #has -1 in locations where nobody agrees (case 1)
    # print(len(ground_truth), sum(ground_truth == -1))
    # print(ground_truth.shape)
    assert shot_predicted_cut.shape[0] == ground_truth_cut.shape[0]
    # print(shot_predicted_cut.shape, ground_truth.shape)
    k_st = k_statistic(shot_predicted_cut.LHD_det.values, ground_truth_cut)
    
    k_st_blocks = k_statistic(states_blocks, ground_truth_blocks)
    
    conf_mat, mat_order = confusion_matrix_blocks(states_blocks, ground_truth_blocks, num_classes=3)
    # plt.matshow(conf_mat)
    # plot_conf_mat(conf_mat, mat_order)
    # plt.imshow()
    # exit(0)
    
    # assert(len(ground_truth) == len(fshot_equalized))
    # fshot_equalized = fshot_equalized.drop(columns=['LHD_label'])
    # print(fshot_equalized)
    # fshot_equalized.loc[:, 'LHD_label'] = ground_truth
    # shot_predicted_cut['LHD_label'] = ground_truth_cut
    # plot_shot_simplified(shot_id, shot_predicted, train_dir+'/'+str(shot_id)+'prediction.pdf')
    print('k stat per time', k_st)
    print('k stat per blocks', k_st_blocks)
    # exit(0)
    return k_st, ground_truth, ground_truth_cut, shot_predicted, shot_predicted_cut, k_st_blocks, ground_truth_blocks, states_blocks, [conf_mat, mat_order] #.LHD_det.values

def decode_context(input_seq, encoder, decoder, num_transitions, block_size, max_source_sentence_chars,
                   conv_w_size, stride, num_channels, max_source_sentence_words, max_train_target_words, look_ahead, decoder_type):
    target_num_blocks = (input_seq.shape[1] - 2*look_ahead) // block_size #should be int
    # print(input_seq.shape, target_num_blocks, num_transitions)
    # exit(0)
    # target_blocks_per_source_sentence = (max_source_sentence_chars - 2*look_ahead) // block_size
    target_blocks_per_source_sentence = max_train_target_words
    # target_blocks_per_source_sentence has shrunk due to look ahead and remaining source characters
    # print(max_source_sentence_chars, target_blocks_per_source_sentence)
    # exit(0)
    decoded_sequence = []
    decoded_sequence_categorical = []
    predicted_transitions = []
    target_seq = np.zeros((1, 1, num_transitions)) #1 START block
    target_seq[0,0,-1] = 1 
    # print('target', target_seq)
    stop = 0
    target_chars_per_source_sentence = target_blocks_per_source_sentence * block_size
    # remainder = max_source_sentence_chars - target_chars_per_source_sentence
    # print(target_blocks_per_source_sentence, target_chars_per_source_sentence)
    # cumul_remainder = 0
    # exit(0)
    attention_weights_sequence = []
    attention_weights = np.zeros((1,1,max_source_sentence_words))
    attention_context = decoder.initialize_context()
    for k in range(target_num_blocks): # - subseq_size//block_size
    # for k in range(250, 270):
      # stop += 1
      # print(k)
        if k % target_blocks_per_source_sentence == 0:
          # decoder_model.reset_states()
            subseq_st_index = k * block_size
            subseq_end_index = subseq_st_index + max_source_sentence_chars
            
            # cumul_remainder += remainder
            subsequence = input_seq[:, subseq_st_index: subseq_end_index, :]
            # print(k, block_size, subseq_st_index, subseq_end_index, subsequence.shape[1], target_chars_per_source_sentence)
            if subsequence.shape[1] < max_source_sentence_chars: #no more full subsequences to predict on
              # print('breaking')
                break
            windowed_subsequence = np.empty((1, int((subsequence.shape[1]-conv_w_size + stride)/stride),conv_w_size, num_channels), dtype=np.float32)
            # print(windowed_subsequence.shape, target_chars_per_source_sentence)
            for l in range(windowed_subsequence.shape[1]):
                windowed_subsequence[:, l] = subsequence[:, stride*l : stride*l+conv_w_size]
            # print(windowed_subsequence.dtype, subsequence.dtype)
            # exit(0)
            encoder_output, states = encoder(windowed_subsequence)
            # states = decoder_model.initialize_hidden_states()
            context = decoder.initialize_context()
        
        if decoder_type == 'attentionless':
            predicted_trans, states = decoder(target_seq, states)
        else:
            predicted_trans, states, attention_weights, attention_context = decoder(target_seq, states, encoder_output, attention_context)
        
      
        # print(predicted_trans.shape, context.shape, states[0].shape, attention_weights.shape)
        # exit(0)
        predicted_trans = np.expand_dims(predicted_trans, 1) # to ensure 1 dimension is the time dimension
        assert(len(predicted_trans.shape) == 3)
        # assert np.round(np.sum(attention_weights[0,0]), 5) == 1
        # print(attention_weights.shape, np.sum(attention_weights[0,0]))
        # exit(0)
        # predicted_trans, _, attention_weights = decoder_model(target_seq, encoder_output)
        
        sampled_transition = np.argmax(predicted_trans[0, 0, :]) #
        # print(k, predicted_trans[0].shape, predicted_trans[0, 0, :], sampled_transition)
        # exit(0)
        decoded_sequence.append(sampled_transition)
        predicted_transitions.extend(predicted_trans)
          
        zeros = np.zeros(num_transitions)
        zeros[sampled_transition] = 1
        decoded_sequence_categorical.append(zeros)
        # print(k,sampled_transition)
        target_seq = np.zeros((1, 1, num_transitions))
        target_seq[0, 0, sampled_transition] = 1
        # print(np.asarray(attention_weights_sequence).shape, attention_weights.shape)
        attention_weights_sequence.extend(attention_weights)
        # if stop ==10:
        #     break
    print(np.asarray(attention_weights_sequence).shape)
    attention_weights_sequence = np.asarray(attention_weights_sequence).swapaxes(1,2)
    decoded_sequence_categorical = np.asarray(decoded_sequence_categorical)
    predicted_transitions = np.asarray(predicted_transitions).swapaxes(0,1)
    decoded_sequence = np.asarray(decoded_sequence).reshape(len(decoded_sequence), 1)
    return decoded_sequence,predicted_transitions,decoded_sequence_categorical,attention_weights_sequence#, target_blocks_per_source_sentence

def beam_search(input_seq, encoder, decoder, num_outputs, block_size, max_source_sentence_chars,
                   conv_w_size, stride, num_channels, max_source_sentence_words, max_train_target_words, look_ahead, beam_width, decoder_type, path_logs, verbose=False):
    target_num_blocks = (input_seq.shape[1] - 2*look_ahead) // block_size #should be int
    print(input_seq.shape, target_num_blocks)
    target_blocks_per_source_sentence = max_train_target_words
    # target_blocks_per_source_sentence has shrunk due to look ahead and remaining source characters
    decoded_sequence_int = []
    decoded_sequence_categorical = []
    predicted_trans_sequence = []
    last_output = np.zeros((1, 1, num_outputs)) #1 START block
    last_output[0,0,-1] = 1 
    target_chars_per_source_sentence = target_blocks_per_source_sentence * block_size
    # print(num_outputs)
    # exit(0)
    attention_weights_sequence = []
    # print(decoder.initialize_context().shape)
    # exit(0)
    # initial_beam = Beam(np.log(1), np.asarray([Transitions.no_trans(),]), np.asarray([States.L()]),
    #                     decoder.initialize_context(), decoder.initialize_hidden_states(), np.empty((1,1,max_source_sentence_words)), num_outputs)
    initial_beam = StateBeam(np.log(1), np.asarray([]), np.asarray([]),
                        decoder.initialize_context(), decoder.initialize_hidden_states(), np.empty((1,1,max_source_sentence_words)), num_outputs)
    beams = [initial_beam,]
    
    # attention_weights = np.zeros((1,1,max_source_sentence_words))
    # attention_context = decoder.initialize_context()
    start = datetime.now()
    if verbose:
        tf.profiler.experimental.start(path_logs)
        start_subseq = datetime.now()
        print('starting beam search', start)
    for k in range(target_num_blocks):
        
      # exit(0)
      # for k in range(30):
        # print('decoding step', k+1, 'number of active beams', len(beams), beams[0])
        # print(time.time())
        new_beams = []
        
        if k % target_blocks_per_source_sentence == 0:
            if verbose:
                prev = start_subseq 
                start_subseq = datetime.now()
                print('-----------------------------------computing new subseq.-----------------------------------')
                print('last subseq. took', (start_subseq - prev).microseconds/1000, 'ms to encode and decode, should take at most', max_source_sentence_chars/10)
            # exit(0)
            #prune search tree of all equal paths when decoder states are reset
            #equal paths are those which have the same last transition, and the same last state
            #in practice, this means a reset of states also resets the number of paths to a maximum of 9
            sorted_prob_beams = sorted(beams, key=attrgetter('probability'))
            # print('new subseq')
            # for b in sorted_prob_beams:
            #   if len(b.plasma_states) > 0:
            #     print(b.probability, b.plasma_states[-1])
            # 
            subseq_st_index = k * block_size
            subseq_end_index = subseq_st_index + max_source_sentence_chars
            
            # cumul_remainder += remainder
            subsequence = input_seq[:, subseq_st_index: subseq_end_index, :]
            # print(k, block_size, subseq_st_index, subseq_end_index, subsequence.shape[1], target_chars_per_source_sentence)
            if subsequence.shape[1] < max_source_sentence_chars: #no more full subsequences to predict on
              # print('breaking')
                break
            if verbose:
                print('fetching windows.,',  (datetime.now() - start_subseq).microseconds/1000)
            windowed_subsequence = np.empty((1, int((subsequence.shape[1]-conv_w_size + stride)/stride),conv_w_size, num_channels), dtype=np.float32)
            for l in range(windowed_subsequence.shape[1]):
                windowed_subsequence[:, l] = subsequence[:, stride*l : stride*l+conv_w_size]
            # print('got windows.,',  (datetime.now() - start).microseconds/1000)
            if verbose:
                print('encoding.,',  (datetime.now() - start_subseq).microseconds/1000)
            
            encoder_output, states = encoder(windowed_subsequence,verbose=verbose)
            
            if verbose:
                print('encoded.,',  (datetime.now() - start_subseq).microseconds/1000)
            # print(decoder.initialize_context().shape)
            # print(beams)
            # exit(0)
            for beam in beams:
                beam.attention_context = decoder.initialize_context()
                beam.hidden_states = states
            # exit(0)
            if verbose:
                print('computed new subseq.,',  (datetime.now() - start_subseq).microseconds/1000, 'this value would ideally be below', max_source_sentence_chars/10)
            # exit(0)
        # print(time.time())
        if verbose:
            print('------------------computing beams------------------')
        beam_start = datetime.now()
        for beam in beams:
            if len(beam.plasma_states) == 0:
                last_state = int(States.L())
            else:
                last_state = int(beam.plasma_states[-1])
            # if k > 0:
             
            # print('last_state', last_state)
            # exit(0)
            last_output = np.zeros((1, 1, num_outputs)) #1 START block
            # last_output[0,0,last_trans] = 1
            last_output[0,0,last_state-1] = 1
            states = beam.hidden_states
            attention_context = beam.attention_context
            if verbose:
                print('decoding.,',  (datetime.now() - beam_start).microseconds/1000)
            if decoder_type == 'attentionless':
                predicted_output, states = decoder(last_output, states)
                attention_weights = []
            else:
                predicted_output, states, attention_weights, attention_context = decoder(last_output, states, encoder_output, attention_context)
            
            if verbose:
                print('decoded.,',  (datetime.now() - beam_start).microseconds/1000)
            # print(predicted_output.shape, attention_context.shape, states[0].shape, attention_weights.shape)
            # exit(0)
            predicted_output = np.expand_dims(predicted_output, 1) # to ensure 1 dimension is the time dimension
            assert(len(predicted_output.shape) == 3)
            # assert np.round(np.sum(attention_weights[0,0]), 5) == 1
      
            
            # sampled_transition = np.argmax(predicted_trans[0, 0, :]) #
            
            
            log_predicted_output = np.log(predicted_output)
            #generate all next paths, choose <beam-width> most likely ones
            # beams_temp = []
            # print(time.time())
            branches = beam.update(log_predicted_output, states, attention_context, attention_weights)
            # print(len(branches))
            # exit(0)
            # print(beam.attention_weights.shape)
            # print(np.sum(beam.attention_weights[k, 0, :]))   
            if k < 1:
                new_beams.append(branches[0])#in first blocks, we can only be in Low
            else:
                new_beams.extend(branches)
            
          
          # print(k, predicted_trans[0].shape, np.mean(decoder_h), np.mean(decoder_c), predicted_trans[0, 0, :], sampled_transition)
          # decoded_sequence_int.append(sampled_transition)
          # predicted_trans_sequence.extend(predicted_trans)
            
          # zeros = np.zeros(num_transitions)
          # zeros[sampled_transition] = 1
          # decoded_sequence_categorical.append(zeros)
    
          # last_output = np.zeros((1, 1, num_transitions))
          # last_output[0, 0, sampled_transition] = 1
          # decoded_sequence_categorical.append(last_output[0,0])
          
          # attention_weights_sequence.extend(attention_weights)
            
        #sort new_beams by highest to lowest prob
        #if len (new_beams) > <beam_width>, eliminate all lowest-prob beams until len becomes <beam_width>
        # print(new_beams[0].transitions)
        # print(new_beams[1].transitions)
        # print(new_beams[2].transitions)
        # exit(0)
        if verbose:
            print('computed beams,',  (datetime.now() - beam_start).microseconds/1000, 'this should be at most', block_size/10)
        # print('sorting beams,',  (datetime.now() - start).microseconds/1000)
        sorted_new_beams = sorted(new_beams, key=attrgetter('probability'), reverse=True)
        # print('sorted beams,',  (datetime.now() - start).microseconds/1000)
        # if len(sorted_new_beams) > beam_width:
        beams = sorted_new_beams[:beam_width]
        
        # beams = sorted_new_beams
        
        #this code to preserve at least 1 beam in each state at each time step
        # beams = []
        # for plasma_state in [States.L(), States.D(), States.H()]:
        #  # beam = next((b for b in sorted_prob_beams if b.plasma_states[-1] == plasma_state), empty_value)
        #  # print(plasma_state)
        #  for b in range(len(sorted_new_beams)):
        #   beam = sorted_new_beams[b]
        #   # print(sorted_new_beams, b)
        #   if beam.plasma_states[-1] == plasma_state:
        #     beams.append(sorted_new_beams.pop(b))
        #     break
        # 
        # while len(beams) < beam_width:
        #   if len(sorted_new_beams) > 0:
        #     beams.append(sorted_new_beams.pop(0))
        #   else:
        #     break
        # if k> 2:
        #     exit(0)
    if verbose:
        tf.profiler.experimental.stop()
    end = datetime.now()
    #print(end)
    delta = end-start
    print('Time taken for beam search', delta)
    # print(encoder.summary())
    # print(decoder.summary())
    return beams





