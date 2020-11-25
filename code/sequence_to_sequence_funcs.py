from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from keras.layers import Input, MaxPooling1D, Flatten, concatenate, Dense, Dropout, BatchNormalization, Conv1D, Activation, RepeatVector, Lambda
from keras.models import Model
from keras.optimizers import Adam
from sequence_to_sequence_data_generator import *
from helper_funcs import load_fshot_from_labeler, normalize_signals_mean, det_trans_to_state, get_trans_ids, load_fshot_from_labeler, calc_mode
from keras import backend as K
import os
import matplotlib as mpl
import tensorflow as tf
from plot_shot_results import plot_shot_full, plot_shot_simplified
from plot_scores import plot_kappa_histogram
# mpl.use('Agg')


# def decode_context(input_seq, encoder_model, decoder_model, num_transitions, block_size, subseq_size, conv_w_size, stride, num_channels):
def decode_context(input_seq, encoder_model, decoder_model, num_transitions, block_size, max_source_sentence_chars, conv_w_size, stride, num_channels, words_per_sentence, look_ahead):
    target_num_blocks = (input_seq.shape[1] - 2*look_ahead) // block_size #should be int
    print(input_seq.shape, target_num_blocks)
    # exit(0)
    target_blocks_per_source_sentence = (max_source_sentence_chars - 2*look_ahead) // block_size
    # target_blocks_per_source_sentence has shrunk due to look ahead and remaining source characters
    decoded_sequence = []
    decoded_sequence_categorical = []
    predicted_transitions = []
    target_seq = np.zeros((1, 1, num_transitions)) #1 START block
    target_seq[0,0,-1] = 1 
    # print('target', target_seq)
    stop = 0
    # print('complete this')
    # exit(0)
    decoder_model.reset_states()
    encoder_model.reset_states()
    target_chars_per_source_sentence = target_blocks_per_source_sentence * block_size
    remainder = max_source_sentence_chars - target_chars_per_source_sentence
    # print(target_chars_per_source_sentence, remainder)
    # cumul_remainder = 0
    # exit(0)
    for k in range(target_num_blocks): # - subseq_size//block_size
    # for k in range(250, 270):
        stop += 1
        # print(k, blocks_in_subseq)
        if k % target_blocks_per_source_sentence == 0:
            # decoder_model.reset_states()
            subseq_st_index = k * block_size
            subseq_end_index = subseq_st_index + max_source_sentence_chars
            
            # cumul_remainder += remainder
            subsequence = input_seq[:, subseq_st_index: subseq_end_index, :]
            # print(k, block_size, subseq_st_index, subseq_end_index, subsequence.shape[1])
            if subsequence.shape[1] < target_chars_per_source_sentence: #no more full subsequences to predict on
                print('breaking')
                break
            windowed_subsequence = np.empty((1, int((subsequence.shape[1]-conv_w_size + stride)/stride),conv_w_size, num_channels))
            # print(windowed_subsequence.shape, target_chars_per_source_sentence)
            for l in range(windowed_subsequence.shape[1]):
                # print(stride*l, stride*l+conv_w_size)
                windowed_subsequence[:, l] = subsequence[:, stride*l : stride*l+conv_w_size]
                
            # exit(0)
            h, c = encoder_model.predict(windowed_subsequence)
        # continue
        predicted_trans, h, c = decoder_model.predict({'decoder_inputs':target_seq,
                                                 'decoder_input_forward_h':h,
                                                 'decoder_input_forward_c':c,
                                                 # 'decoder_input_past_h':encoder_h_past,
                                                 # 'decoder_input_past_c':encoder_c_past,
                                                 })#, 'decoder_state_input_h':d_h, 'decoder_state_input_c':d_c} )
        # print(predicted_trans.shape, h.shape, c.shape)
        # encoder_h_past, encoder_c_past = decoder_h, decoder_c
        # print(predicted_trans[0])
        # exit(0)
        sampled_transition = np.argmax(predicted_trans[0, 0, :])
        # print(k, predicted_trans[0].shape, np.mean(decoder_h), np.mean(decoder_c), predicted_trans[0, 0, :], sampled_transition)
        # if k < 100:
        #     sampled_transition = 6
        decoded_sequence.append(sampled_transition)
        predicted_transitions.extend(predicted_trans)
        
        zeros = np.zeros(num_transitions)
        zeros[sampled_transition] = 1
        decoded_sequence_categorical.append(zeros)

        target_seq = np.zeros((1, 1, num_transitions))
        target_seq[0, 0, sampled_transition] = 1
        
        # if stop ==10:
        #     break
    # exit(0)
    return np.asarray(decoded_sequence).reshape(len(decoded_sequence), 1), np.asarray(predicted_transitions).swapaxes(0,1), np.asarray(decoded_sequence_categorical)


    
def train(train_dir, conv_w_size, no_input_channels, latent_dim, num_transitions, num_epochs, train_generator, val_generator, epoch_size, bsize):
    conv_input = Input(shape=(conv_w_size,no_input_channels,), dtype='float32', name='conv_input')
    
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(conv_input)
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(x_conv)
    # x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(x_conv)
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(x_conv)
    # x_conv = Dropout(.5)(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    # x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv)
    # x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv)
    # x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv)
    # # # x_conv = Dropout(.5)(x_conv)
    # x_conv = MaxPooling1D(2)(x_conv)
    conv_out = Flatten()(x_conv)
    conv_out = Dense(32, activation='relu', name='scalar_output')(conv_out)
    # conv_out = Dropout(.5)(conv_out)
    conv_out = Dense(32, activation='relu')(conv_out)
    # conv_out = Dropout(.5)(conv_out)
    modelCNN = Model(inputs=[conv_input], outputs= [conv_out])
    print(modelCNN.summary())
    
    encoder_inputs = Input(shape=(None,conv_w_size,no_input_channels), dtype='float32', name='encoder_inputs')
    encoder_t_dist = TimeDistributed(modelCNN)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm') # Bidirectional()
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_t_dist)
    # encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))(encoder) # Bidirectional()
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_transitions), name='decoder_inputs')
    # decoder_past_input_h = Input(shape=(latent_dim,), name='decoder_past_input_h')
    # decoder_past_input_c = Input(shape=(latent_dim,), name='decoder_past_input_c')
    # decoder_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
    # decoder_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
    # 
    # def repeat_vector(args):
    #     encoder_states = args[0]
    #     decoder_inputs = args[1]
    #     return RepeatVector(K.shape(decoder_inputs)[1])(encoder_states)
    # concat_states = concatenate(encoder_states)
    # decoder_context = Lambda(repeat_vector, output_shape=(None, K.int_shape(concat_states)[1])) ([concat_states, decoder_inputs])
  
    # decoder_concat_inputs = concatenate([decoder_inputs, decoder_context])
    # dense_decoder_input = TimeDistributed(Dense(latent_dim, activation='relu'))#
    # timedist_decoder_input = dense_decoder_input(decoder_concat_inputs)
    # timedist_decoder_input = dense_decoder_input(decoder_inputs)
    # exit(0)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    # print(decoder_lstm.stateful)
    # exit(0)
    # decoder_outputs, _, _ = decoder_lstm(timedist_decoder_input)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_tdist = TimeDistributed(Dense(32, activation='relu'))
    decoder_outputs = decoder_tdist(decoder_outputs)
    
    decoder_dense = TimeDistributed(Dense(num_transitions, activation='softmax'), name='decoder_outputs')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # encoder_decoder = Model([encoder_inputs, decoder_inputs, decoder_past_input_h, decoder_past_input_c], decoder_outputs)
    encoder_decoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(encoder_decoder.summary())
    
    # encoder = Model(encoder_inputs, encoder_states)
    
    # gen_val = next(iter(val_generator))
    # gen_train = next(iter(train_generator))
    encoder_decoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # encoder_decoder.fit_generator(generator = gen_train, steps_per_epoch=epoch_size, epochs=num_epochs, validation_data=gen_val, validation_steps=bsize)
    # train_context(encoder_decoder, encoder, train_generator, num_epochs, epoch_size, val_generator, bsize)
    encoder_decoder.save(train_dir + '/encoder_decoder.h5')
    # exit(0)
    
    
    # 
    # encoder_inputs = Input(batch_shape=(1,None,conv_w_size,no_input_channels), dtype='float32', name='encoder_inputs')
    # encoder_t_dist = TimeDistributed(modelCNN)(encoder_inputs)
    # weights = encoder_decoder.get_layer('encoder_lstm').get_weights()
    # 
    # # print(len(weights))
    # # 
    # # print('-------------------------')
    # # print(np.mean(weights[0]))
    # # print(np.mean(weights[1]))
    # # print(np.mean(weights[2]))
    # # 
    # encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, stateful=True, name='encoder_lstm')
    # encoder_outputs, state_h, state_c = encoder_lstm(encoder_t_dist)
    # encoder_lstm.set_weights(weights)
    #  #, initial_state=decoder_states_inputs
    # encoder_states = [state_h, state_c]
    
    encoder = Model(encoder_inputs, encoder_states)
    encoder.save(train_dir + '/encoder.h5')
    
    
    decoder_input_forward_h = Input(shape=(latent_dim,), name='decoder_input_forward_h')
    decoder_input_forward_c = Input(batch_shape=(1, latent_dim,), name='decoder_input_forward_c')
    decoder_context_inputs = [decoder_input_forward_h, decoder_input_forward_c]
    decoder_inputs = Input(batch_shape=(1, None, num_transitions), name='decoder_inputs')
    # decoder_input_past_h = Input(batch_shape=(1, latent_dim,), name='decoder_input_past_h')
    # decoder_input_past_c = Input(batch_shape=(1, latent_dim,), name='decoder_input_past_c')
    # decoder_past_inputs = [decoder_input_past_h, decoder_input_past_c]
    
    
    
    # concat_states = concatenate(decoder_context_inputs)
    # decoder_context = Lambda(repeat_vector, output_shape=(None, K.int_shape(concat_states)[1])) ([concat_states, decoder_inputs])
    # decoder_concat_inputs = concatenate([decoder_inputs, decoder_context])
    # timedist_decoder_input = dense_decoder_input(decoder_concat_inputs)
    # timedist_decoder_input = dense_decoder_input(decoder_inputs)
    
    # weights = encoder_decoder.get_layer('decoder_lstm').get_weights()
    # print(np.mean(decoder_lstm.get_weights()[1]))
    # decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm') #stateful=True, 
    decoder_outputs, decoder_h, decoder_c = decoder_lstm(decoder_inputs, initial_state=decoder_context_inputs) #, initial_state=decoder_states_inputs
    # print(np.mean(decoder_lstm.get_weights()[1]))
    # decoder_lstm.set_weights(weights)
    # exit(0)
    decoder_outputs = decoder_tdist(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # decoder = Model([decoder_inputs] + decoder_context_inputs + decoder_past_inputs, [decoder_outputs, decoder_h, decoder_c])
    decoder = Model([decoder_inputs] + decoder_context_inputs, [decoder_outputs, decoder_h, decoder_c])
    decoder.save(train_dir + '/decoder.h5')
    
    # print(len(weights))
    # 
    # print('-------------------------')
    # print(np.mean(weights[0]))
    # print(np.mean(weights[1]))
    # print(np.mean(weights[2]))
    # 
    decoder.compile(optimizer='adam', loss='categorical_crossentropy')
    print('exiting')
    exit(0)
    
    return encoder_decoder, encoder, decoder
    # exit(0)
def predict(data_dir, train_dir, labelers, num_transitions, block_size, max_source_sentence_chars, conv_w_size, stride, latent_dim, num_channels, val_shots, max_source_sentence_words, look_ahead, pred_start, pred_interval):
    encoder_model = load_model(train_dir + '/encoder.h5')
    decoder_model = load_model(train_dir + '/decoder.h5')
    
    k_indexes = []
    for shot in val_shots:
        # print(shot)
        # pred_start = 100
        # pred_interval = 800
        shot_df, fshot_times = load_fshot_from_number(shot, data_dir)
        
        # start = 2250   #LH transition @ 2450
        shot_df = normalize_signals_mean(shot_df)[pred_start:pred_start+pred_interval]
        shot_df = shot_df.drop(columns=['LHD_label', 'ELM_label'])
        # print(shot_df.shape)
        shot_signals = get_raw_signals_in_window(shot_df).swapaxes(0,1)
        # plt.plot(shot_signals)
        # plt.show()
        # shot_df = state_to_trans_event_disc(shot_df, gaussian_hinterval)
        # shot_signals = get_transitions_in_window(shot_df)
        # print('USING IDENTITY, CAREFUL shot_signals', shot_signals.shape)
        
        # look_ahead=5
        # decoded_sequence, timewise_preds, timewise_preds_cat = decode_context(
            # shot_signals.reshape(1, shot_signals.shape[0], shot_signals.shape[1]), encoder_model, decoder_model, num_transitions,
            # block_size, timesteps, conv_w_size, stride, no_input_channels)
        decoded_sequence, timewise_preds, timewise_preds_cat = decode_context(shot_signals.reshape(1, shot_signals.shape[0], shot_signals.shape[1]),
                       encoder_model, decoder_model, num_transitions, block_size,
                       max_source_sentence_chars, conv_w_size, stride, num_channels, max_source_sentence_words, look_ahead)
        
        # decoded_sequence, timewise_preds = decode_context_decoder_stateful(
        #     shot_signals.reshape(1, shot_signals.shape[0], shot_signals.shape[1]), encoder_model, decoder_model, num_transitions, block_size, timesteps, conv_w_size, stride, no_input_channels)
        # print(decoded_sequence.shape, timewise_preds.shape, timewise_preds_cat.shape)
        
        # exit(0)
        
        shot_cut = shot_df[look_ahead: ] #look_ahead +
        # print(shot_df.shape, shot_signals.shape, shot_la_cut.shape)
        # exit(0)
        timewise_preds_cat = np.repeat(timewise_preds_cat, block_size, axis=0)
        to_cut = len(shot_cut) - len(timewise_preds_cat) 
        # print(shot_la_cut.shape, len(timewise_preds_cat), to_cut)
        shot_cut = shot_cut[:len(shot_cut) - to_cut]
        # print(shot_la_cut.shape)
        # exit(0)
        # shot_cut = shot_cut.loc[:len(timewise_preds_cat)*block_size-1]
        # print(shot_cut.shape)
        shot_cut['ELM_prob'] = np.zeros(len(shot_cut))
    
        for t, t_id in enumerate(trans_ids):
            # fshot[t_id + '_det_prob'] = pred_trans[:, t]
            # print(len( np.repeat(timewise_preds_cat[:, t], block_size)[:len(shot_cut)]))
            shot_cut[t_id + '_det'] = timewise_preds_cat[:, t]
        # exit(0)
        # print(shot_cut)
        shot_lhd_det = det_trans_to_state(shot_cut)
        shot_cut['LHD_det'] = shot_lhd_det
        # k_st = k_statistic(shot_lhd_det, shot_la_cut['LHD_label'].values)
        # k_indexes += [k_st]
        # print(k_st)
        # # print(shot_lhd_det)
        # # exit(0)

        # np.save(train_dir+'/'+str(shot)+'decoded.npy', decoded_sequence)
        # np.save(train_dir+'/'+str(shot)+'timewise.npy', timewise_preds)
        
        k_st = evaluate(shot, data_dir, train_dir, labelers, shot_cut)
        k_indexes += [k_st]
        
        
    # k_indexes = np.expand_dims(np.asarray(k_indexes), 1)
    
    
    print(k_indexes)
    # exit(0)
    k_indexes = np.asarray(k_indexes)
    histo_fname = train_dir+ '/k_ind_histogram.pdf'
    title = ''
    plot_kappa_histogram(k_indexes, histo_fname, title)
    
def evaluate(shot_id, data_dir,train_dir, labelers, shot_predicted):
    print('Evaluating shot', shot_id)
    intersect_times = np.round(shot_predicted.time.values,5)
    labeler_states = []
    for labeler in labelers:
        shot_id_lab = str(shot_id) + '-' + labeler
        fshot_labeled, fshot_times = load_fshot_from_labeler(shot_id_lab, data_dir)
        intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
        fshot_equalized = (fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]).copy()
        labeler_states += [fshot_equalized['LHD_label'].values]
        
    # print(len(intersect_times))
    # print(fshot_equalized.columns)
    # print(shot_predicted.columns)
    labeler_states = np.asarray(labeler_states)
    # print(labeler_states.shape)
    ground_truth = calc_mode(labeler_states.swapaxes(0,1))
    # print(ground_truth.shape)
    k_st = k_statistic(shot_predicted.LHD_det.values, ground_truth)
    
    assert(len(ground_truth) == len(fshot_equalized))
    # fshot_equalized = fshot_equalized.drop(columns=['LHD_label'])
    # print(fshot_equalized)
    # fshot_equalized.loc[:, 'LHD_label'] = ground_truth
    shot_predicted['LHD_label'] = ground_truth
    plot_shot_simplified(shot_id, shot_predicted, train_dir+'/'+str(shot_id)+'prediction.pdf')
    print(k_st)
    return k_st
    # exit(0)
    # 
    # for k, labeler in enumerate(labelers):
    #     fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
    #     intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
        
    # fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
    # intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
    # intersect_times_d[shot] = intersect_times