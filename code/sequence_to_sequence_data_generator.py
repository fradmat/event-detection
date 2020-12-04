from lstm_data_generator import *
from label_smoothing import get_trans_ids
from os import listdir
from window_functions import get_states_categorical


class SequenceToSequenceStateGenerator(LSTMDataGenerator):
    def __init__(self, shot_ids=[], batch_size=16, n_classes=7, shuffle=True,
                 lstm_time_spread=200, epoch_size=20700, train_data_name = '', conv_w_size=40, no_input_channels = 4,
                 gaussian_hinterval=10, no_classes=3, stride=1, labelers = [],conv_w_offset=0, fixed_inds=None, block_size=10,
                 latent_dim = None, num_epochs = None, data_dir=None, source_words_per_sentence=8, target_words_per_sentence = 8,
                 signal_sampling_rate=1e4, look_ahead=10, machine_id = 'tcv', normalize_per_shot=True,):

        self.block_size = block_size
        self.train_subsequence_size = lstm_time_spread
        self.train_block_subseq_size = self.train_subsequence_size // self.block_size
        self.n_classes = n_classes
        super().initialize(shot_ids, batch_size, n_classes, shuffle,lstm_time_spread, epoch_size,
                        train_data_name, conv_w_size, no_input_channels,gaussian_hinterval,
                        no_classes, stride, labelers,conv_w_offset, fixed_inds, signal_sampling_rate, machine_id, normalize_per_shot, data_dir)
        # exit(0)
        self.machine_id = machine_id
        # print('initialized.')
        self.first_prepro_cycle()
        # print(len(self.shot_dfs.keys()), self.shot_dfs.keys())
        # print(len(self.intc_times.keys()), sorted(self.intc_times.keys()))
        # exit(0)
        # print('finished first prepro cycle')
        super().sec_prepro_cycle()
        # print('finished second prepro cycle')
        
        # self.block_dfs = {} #blocks matching shot labelings
        self.source_numbers_words = source_words_per_sentence
        self.target_numbers_words = target_words_per_sentence
        self.trans_samples = {key: [] for key in self.source_numbers_words}
        self.non_trans_samples = {key: [] for key in self.source_numbers_words}
        self.look_ahead = look_ahead
        
        self.third_prepro_cycle()
        samples_per_type = batch_size
        # print(self.norm_factors)
        # exit(0)
        
    
    def first_prepro_cycle(self,):
        gaussian_hinterval = self.gaussian_hinterval
        count = 0
        for shot in self.shot_ids:
            # print(self.machine_id, self.data_dir, shot)
            # exit(0)
            fshot, fshot_times = load_fshot_from_labeler(shot, self.machine_id, self.data_dir)
            # print(fshot)
            # exit(0)
            # fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values_gauss(fshot.LHD_label.values,
            #                                                                                                                                     fshot.time.values,
            #                                                                                                                                     smooth_window_hsize=gaussian_hinterval)
            #                    
            try:            
            # fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
                fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values_gauss(fshot.LHD_label.values,
                                                                                                                                                fshot.time.values,
                                                                                                                                                smooth_window_hsize=gaussian_hinterval)
                #CUTOFF to put all elm labels at 0 where state is not high
                fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0
                
            except:
                print('problem processing shot', shot, ', Try shortening the size of the preprocessing smoothing window (gaussian_hinterval).')    
                exit(0)
            fshot = state_to_trans_event_disc(fshot, gaussian_hinterval)
            # fshot = trans_disc_to_cont(fshot, gaussian_hinterval)
            self.shot_dfs[str(shot)] = fshot.copy()
            count += 1
            # exit(0)
        print('read', str(count), 'shot files.')
    
    def third_prepro_cycle(self,):
        # for shot_id in self.shot_ids: #shot_id refers not just to a shot, but also to a labeler of that shot!
        for shot_id in self.shot_dfs.keys():
            # print(shot_id)
            fshot = self.shot_dfs[str(shot_id)]
            signal_sequence = get_raw_signals_in_window(fshot).swapaxes(0,1)
            times = fshot.time.values
            start_block = np.zeros((self.block_size, self.n_classes))
            # transitions = np.concatenate([start_block, get_transitions_in_window(fshot)])
            # print(fshot.LHD_label.values.shape, start_block.shape)
            states = np.concatenate([start_block, get_states_categorical(fshot, self.n_classes)])
            # print(len(fshot), signal_sequence.shape, times.shape, states.shape)
            # plot_all_signals_states(np.zeros((len(fshot), 2)), signal_sequence, times, states[self.block_size:])
            
            # print(states.shape)
            # exit(0)
            for source_num_words, target_num_words in zip(self.source_numbers_words, self.target_numbers_words):
                source_num_chars = self.stride * (source_num_words - 1) + self.conv_w_size
                # print(source_num_chars)
                subseqs_in_shot = len(fshot) - source_num_chars - self.look_ahead + 1
                for source_ind in range(self.look_ahead, subseqs_in_shot, 1): #, self.train_subsequence_size self.block_size, 1
                    # source_start = source_ind - self.look_ahead
                    # source_end = source_ind + source_num_chars + self.look_ahead
                    # print(source_num_chars, look_ahead_source_start, look_ahead_source_end)
                    # exit(0)
                    signal_subsequence = signal_sequence[source_ind : source_ind + source_num_chars]
                    times_subsequence = times[source_ind : source_ind + source_num_chars]
                    times_subsequence_disloc = times[source_ind - self.block_size: source_ind + source_num_chars]
                    # there might be a minor bug (non-critical) in the line above
                    # matching dislocated times for transitions (which have been dislocated to the future by 1 block)
                    #we need an additional block_size index at the end to fetch the shifted block for teaching forcing during training
                    target_subsequence = states[source_ind + self.look_ahead : source_ind + source_num_chars + self.block_size - self.look_ahead]
                    # print(signal_subsequence.shape, target_subsequence.shape, times_subsequence.shape, times_subsequence_disloc.shape)
                    # exit(0)   
                    block_sequence = np.zeros((target_num_words + 1, self.n_classes)) #+1 for shifted sequence
                    # shifted_block_sequence = np.zeros((target_num_words, self.n_classes))
                    target_num_chars = self.block_size * target_num_words
                    # if target_num_chars > (len(target_subsequence) - self.block_size):
                    # print(target_subsequence.shape, block_sequence.shape)
                    # exit(0)
                    if source_num_chars < target_num_chars + 2 * self.look_ahead: #+ self.block_size 
                        print(source_num_chars, target_num_chars + self.block_size + 2 * self.look_ahead)
                        print('Target sequences must be (in terms of total time slices/chars,', target_num_chars, '),')
                        print('shorter than or equal to the source sequences, (len=', len(signal_subsequence), 'timeslices/chars)')
                        print('minus the forward and backward look-ahead (', self.look_ahead, 'each)')
                        print('i.e., shorter than or equal to', len(signal_subsequence)-2*self.look_ahead, '.') #-self.block_size, 
                        print('Either decrease the number of target words/blocks (or their size),')
                        print('Or, increase the number of source words/conv. windows (or their size), or their convolutional stride.')
                        exit(0)
                    for k,l in enumerate(range(0, target_num_chars + self.block_size, self.block_size)):
                        block = np.zeros((1, self.n_classes))
                        assert l + self.block_size <= len(target_subsequence)
                        block_transitions = target_subsequence[l : l + self.block_size]
                        # print(block_transitions, block_transitions.shape, np.argmax(np.sum(block_transitions, axis=0)))
                        block[:,np.argmax(np.sum(block_transitions, axis=0))] = 1
                        block_sequence[k] = block
                        
                    # print(target_num_chars, self.block_size)
                    # print(target_subsequence[:target_num_chars + self.block_size,].shape)
                    # block_sequence = (target_subsequence[:target_num_chars + self.block_size])[::self.block_size]
                    
                    # print(block_sequence.shape)
                    # exit(0)
                    shifted_block_sequence = block_sequence[1:]
                    block_sequence = block_sequence[:-1]
                    # trans_in_block = (np.sum(shifted_block_sequence[:, :-1])>=1).astype(float).astype(bool)
                    # trans_in_block = (np.sum(shifted_block_sequence[:, :-1])>=1).astype(float).astype(bool)
                    trans_in_block = False
                    if np.count_nonzero(np.sum(shifted_block_sequence, axis=0)) > 1:
                        trans_in_block = True
                    # print(trans_in_block)
                    # exit(0)
                    # print(k, k + self.train_subsequence_size, k//self.train_block_subseq_size, k//self.train_block_subseq_size +self.train_block_subseq_size)
                    # print(signal_sequence.shape, block_sequence.shape)
                    # windowed_signal = np.empty((int((self.train_subsequence_size-self.conv_w_size + self.stride)/self.stride), self.conv_w_size, self.no_input_channels))
                    windowed_signal = np.empty((source_num_words, self.conv_w_size, self.no_input_channels))
                    # print(windowed_signal.shape)
                    # exit(0)
                    # windowed_signal = np.empty((int((self.train_subsequence_size-self.conv_w_size + self.stride)/self.stride), self.conv_w_size, self.no_input_channels))
                    for m in range(len(windowed_signal)):
                        # print(m*self.stride)
                        windowed_signal[m] = signal_subsequence[m*self.stride : m*self.stride+self.conv_w_size] #2:3
                        
                    # past_windowed_signal = np.empty((int((self.train_subsequence_size-self.conv_w_size + self.stride)/self.stride), self.conv_w_size, self.no_input_channels))
                    # # windowed_signal = np.empty((int((self.train_subsequence_size-self.conv_w_size + self.stride)/self.stride), self.conv_w_size, self.no_input_channels))
                    # for m in range(len(past_windowed_signal)):
                        # past_windowed_signal[m] = past_signal_subsequence[m*self.stride : m*self.stride+self.conv_w_size] #2:3
                    
                    if trans_in_block:
                        self.trans_samples[source_num_words].append({'input': [windowed_signal,], 'output': [block_sequence, shifted_block_sequence], 'control':[times_subsequence, shot_id]})
                        # plot_windows_blocks(windowed_signal, block_sequence, shifted_block_sequence, times_subsequence, shot_id, self.stride, self.conv_w_size, self.block_size)
                    else:
                        # print(times_subsequence[0], times_subsequence[-1], windowed_signal[0, 0, 2])
                        self.non_trans_samples[source_num_words].append({'input': [windowed_signal,], 'output': [block_sequence, shifted_block_sequence], 'control':[times_subsequence, shot_id]})
                        # plot_windows_blocks(windowed_signal, block_sequence, shifted_block_sequence, times_subsequence, shot_id, self.stride, self.conv_w_size, self.block_size)
                random.shuffle(self.trans_samples[source_num_words])
                random.shuffle(self.non_trans_samples[source_num_words])
        # print(self.trans_samples, self.non_trans_samples)
        print('Data generator ready. ')
        # exit(0)
            
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length
    
    def __getitem__(self, index):
        
        # print 'Generate one batch of data', len(self.sub_generators)
        batch_X_scalars, batch_y_seq, batch_control = [], [], []
        batch = []
        # for sub_generator in self.sub_generators:
            # print('fixed_inds', self.fixed_inds)
            # exit()
            # X_scalars, y_block_trans, y_ts, y_sh = sub_generator[[index, self.fixed_inds]]
        # batch = self.trans_samples[:self.batch_size//2]
        # print(len(self.trans_samples))
        # print(len(self.trans_samples))
        # exit(0)
        # print(self.trans_samples[0])
        batch_timestep_size = np.random.choice(self.source_numbers_words)
        for k in range(self.batch_size//2): #//2
            # print(len(batch))
            sample = [self.trans_samples[batch_timestep_size].pop(0)]
            batch.extend(sample)
            self.trans_samples[batch_timestep_size].extend(sample)
            sample = [self.non_trans_samples[batch_timestep_size].pop(0)]
            self.non_trans_samples[batch_timestep_size].extend(sample)
            batch.extend(sample)
        # exit(0)
        # batch.extend(self.non_trans_samples[:self.batch_size//2])
        # X_scalars, y_block_trans, y_ts, y_sh
        # print(len(batch))
        # 
        # # exit(0)
        for k in range(len(batch)):
            X_scalars, y_block_trans, y_control = batch[k]['input'], batch[k]['output'], batch[k]['control']
            batch_X_scalars += [X_scalars]
            batch_y_seq += [y_block_trans]
            batch_control += [y_control]
            # print(len(batch_X_scalars))
                
        batch_X_scalars = np.asarray(batch_X_scalars)
        batch_y_seq = np.asarray(batch_y_seq)
        batch_control = np.asarray(batch_control)
        # print(batch_X_scalars.shape, batch_y_seq.shape, batch_control.shape)
        # exit(0)
        aux = list(zip(
                        np.asarray(batch_X_scalars),
                        np.asarray(batch_y_seq),
                        np.asarray(batch_control),
                        ))
        # random.shuffle(aux)
        batch_X_scalars, batch_y_seq, batch_control = zip(*aux)
        batch_y_seq = np.asarray(batch_y_seq)
        # print(np.asarray(batch_y_seq).shape)
        # print(np.sum(batch_y_seq, axis=-1))
        # checksum = np.ones(batch_y_seq.shape[:-1])
        # print(np.sum(batch_y_seq, axis=-1))
        # assert np.array_equal(checksum, np.sum(batch_y_seq, axis=-1))
        # print(np.asarray(batch_X_scalars)[:, 0].shape)
        # exit(0)
        return (
                {
                    'encoder_inputs':np.asarray(batch_X_scalars, dtype=np.float32)[:, 0], #[:, :, 0, :]
                    # 'past_encoder_inputs':np.asarray(batch_X_scalars)[:, 1],
                    'decoder_inputs': np.asarray(batch_y_seq, dtype=np.float32)[:, 0]
                },
                {
                    
                    'decoder_outputs':np.asarray(batch_y_seq, dtype=np.float32)[:, 1],
                    'control': np.asarray(batch_control)
                },
                # {
                #     'control': np.asarray(batch_control)
                #     
                # }
                )
    
    def on_epoch_end(self):
        # print("Epoch finished, reshuffling...")
        for val in self.source_numbers_words:
            random.shuffle(self.trans_samples[val])
            random.shuffle(self.non_trans_samples[val])
        pass
    
def main():
    stateful = False
    compress = True
    randomized_compression = False
    gaussian_time_window = 5e-4
    signal_sampling_rate = 1e4
    convolutional_stride = int(10)
    conv_w_size = 40
    # lstm_time_spread = int(256)
    source_words_per_sentence = [27]
    target_words_per_sentence = [18]
    labelers = ['dummy',]
    shuffle=True
    block_size=10
    look_ahead = 60
    params_lstm_random = {
              'batch_size': 128,
              'n_classes': 3,
              # 'lstm_time_spread': lstm_time_spread,
              'epoch_size': 64,
              'train_data_name': 'endtoendrandomizedshot',
            'no_input_channels' : 4,
            'conv_w_size':conv_w_size,
            'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
            'stride':convolutional_stride,
            'labelers':labelers,
            'shuffle':shuffle,
            'block_size':block_size,
            'source_words_per_sentence': source_words_per_sentence,
            'target_words_per_sentence':target_words_per_sentence,
            'look_ahead':look_ahead,
            'signal_sampling_rate': signal_sampling_rate,
            'machine_id': 'DUMMY_MACHINE',
            'data_dir': '../data/DUMMY_MACHINE/'
            # 'fixed_inds' : ["48580-ffelici/18800"]
            }
   
    all_shot_fnames = listdir('../data/DUMMY_MACHINE/dummy')
    # for l in all_shot_fnames:
    #     print(l)
    all_shots = []
    [all_shots.append(shot[14:19]) for shot in all_shot_fnames]
    # all_shots=(64820,)
    # print(all_shots, len(all_shots))
    # exit(0)
    training_generator = SequenceToSequenceStateGenerator(shot_ids=all_shots, **params_lstm_random)
    # training_generator.on_epoch_end()
    # print(training_generator.length)
    # exit(0)
    gen = next(iter(training_generator))
    conv_windows = []
    past_conv_windows = []
    block_sequences = []
    shifted_block_sequences= []
    sequence_shots = []
    sequence_times = []
    counter = 0
    for batch in gen:
        # print(batch[0].keys(), batch[1].keys())
        encoder_inputs = batch[0]['encoder_inputs']
        # past_encoder_inputs = batch[0]['past_encoder_inputs']
        decoder_inputs = batch[0]['decoder_inputs']
        decoder_outputs = batch[1]['decoder_outputs']
        control = batch[1]['control']
        # print(encoder_inputs)
        for sample in range(params_lstm_random['batch_size']):
            conv_windows += [encoder_inputs[sample]]
            # past_conv_windows += [past_encoder_inputs[sample]]
            block_sequences += [decoder_inputs[sample]]
            shifted_block_sequences += [decoder_outputs[sample]]
            sequence_shots += [control[sample][1]]
            sequence_times += [control[sample][0]]
            
           
            if np.any(conv_windows == float('nan')):
                break
        source_num_chars = convolutional_stride * (encoder_inputs.shape[1] - 1) + conv_w_size
        print(encoder_inputs.shape, decoder_outputs.shape, source_num_chars)
        counter += 1
        if counter == params_lstm_random['batch_size']:
            break
        # break
    
    # exit(0)
    conv_windows = np.asarray(conv_windows)
    # past_conv_windows = np.asarray(past_conv_windows)
    block_sequences = np.asarray(block_sequences)
    sequence_times = np.asarray(sequence_times)
    sequence_shots = np.asarray(sequence_shots)
    shifted_block_sequences = np.asarray(shifted_block_sequences)
    # overlap = conv_w_size - convolutional_stride
    # source_num_chars = convolutional_stride * conv_windows.shape[1] + overlap * (conv_windows.shape[1] - 1)

    # source_num_chars = convolutional_stride * (conv_windows.shape[1] - 1) + conv_w_size
    # print('conv_windows:', conv_windows.shape, 'based on', source_num_chars, 'time slices. block_sequences:', block_sequences.shape, shifted_block_sequences.shape, sequence_times.shape, sequence_shots.shape)
    # print(conv_windows.shape)
    # exit(0)
    # , past_conv_windows[k]
    for k in range(params_lstm_random['batch_size']):
        plot_windows_blocks_states(conv_windows[k], block_sequences[k], shifted_block_sequences[k], sequence_times[k], sequence_shots[k], convolutional_stride, conv_w_size, block_size, look_ahead)
        # source_num_chars = convolutional_stride * (conv_windows[k].shape[1] - 1) + conv_w_size
        # print('batch with -> conv_windows:', conv_windows[k].shape, 'based on', source_num_chars, 'time slices. block_sequences:', block_sequences[k].shape, shifted_block_sequences[k].shape, sequence_times.shape, sequence_shots.shape)
        
        
if __name__ == '__main__':
    main()