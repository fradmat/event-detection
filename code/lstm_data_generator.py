import numpy as np
import keras
import pandas as pd
import abc
from window_functions import *
from label_smoothing import *
from helper_funcs import normalize_signals_whole_set, normalize_signals_mean
from os import *
# from plot_shots_and_events import get_window_plots_nn_data
import math
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import glob
# from cnn_data_generator import CNNAugmentedDataFetcher, CNNRandomDataFetcher, IDsAndLabelsCNN
from plot_routines import *

class IDsAndLabels(object):
    def __init__(self,):
        self.ids = {}
        self.len = 0
    
    def __len__(self,):
        return self.len
    
    def generate_id_code(self, shot, index):
        return str(str(shot)+'/'+str(index))
    
    def add_id(self, shot, k, transitions, elms, dithers):
        code = self.generate_id_code(shot, k)
        if code in self.ids.keys():
            return
        else:
            self.ids[code] = {'transitions': transitions, 'elms':elms, 'dithers': dithers}
            self.len += 1
            # print('added id', self.len, len(self.ids))
    
    def get_sorted_ids(self):
        return sorted(self.ids.keys(), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
    
    def get_ids(self):
        # print 'getting ids', self.len, len(self.ids), len(self.ids.keys())
        return list(self.ids.keys())
    
    def get_shot_and_id(self, ID):
        s_i = ID.split('/')
        return s_i[0], int(s_i[1])
            
    def get_label(self, ID):
        return self.ids[ID]
    
    def get_ids_and_labels(self):
        # pairs = np.empty((self.len, 2))
        pairs = []
        sorted_ids = sorted(self.get_ids(), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
        for ind in sorted_ids:
            pairs += [[ind, self.get_label(ind)]]
        return pairs

    def get_shots(self):
        shots = []
        for ID in self.get_sorted_ids():
            shot, ind = self.get_shot_and_id(ID)
            shots += [str(shot)]
        return set(shots)
    
class IDsAndLabelsLSTM(IDsAndLabels):
    def __init__(self,):
        IDsAndLabels.__init__(self)
    
    def add_id(self, shot, k, elm_lab_in_dt, state_lab_in_dt): #state
        code = self.generate_id_code(shot, k)
        self.ids[code] = (elm_lab_in_dt, state_lab_in_dt)
        self.len += 1
        # print('added id', code)
        
class LSTMDataGenerator():
    def __init__(self, shot_ids=[], batch_size=16, n_classes=7, shuffle=True,
                 lstm_time_spread=150, epoch_size=20700, train_data_name = '', conv_w_size=40, no_input_channels = 3,
                 gaussian_hinterval=10, no_classes=3, stride=1, labelers = [],conv_w_offset=20, fixed_inds=None, signal_sampling_rate=1e4, machine_id='TCV',
                 normalize_per_shot=True, data_dir = './labeled_data/TCV/'):

        
        
        self.initialize(shot_ids, batch_size, n_classes, shuffle,lstm_time_spread, epoch_size,
                        train_data_name, conv_w_size, no_input_channels,gaussian_hinterval,
                        no_classes, stride, labelers,conv_w_offset, fixed_inds, signal_sampling_rate, machine_id, normalize_per_shot, data_dir)
        self.first_prepro_cycle()
        self.sec_prepro_cycle()
        self.third_prepro_cycle()
     
        
        print(len(self.ids_low), len(self.ids_dither), len(self.ids_high))
        # exit(0)
        
        samples_per_type = batch_size//4
        self.high_generator = LSTMRandomDataFetcherEndtoEndWOffset('High',
                                                 self.ids_high,
                                                 self.data_dir,
                                                 self.lstm_time_spread, 
                                                samples_per_type,
                                                n_classes,
                                                self.shuffle,
                                                self.conv_w_size,
                                                self.no_input_channels,
                                                self.shot_dfs,
                                                self.no_classes,
                                                self.stride,
                                                self.conv_w_offset)
        self.dither_generator = LSTMRandomDataFetcherEndtoEndWOffset('Dither',
                                                 self.ids_dither,
                                                 self.data_dir,
                                                 self.lstm_time_spread, 
                                                samples_per_type,
                                                n_classes,
                                                self.shuffle,
                                                self.conv_w_size,
                                                self.no_input_channels,
                                                self.shot_dfs,
                                                self.no_classes,
                                                self.stride,
                                                self.conv_w_offset)
        self.low_generator = LSTMRandomDataFetcherEndtoEndWOffset('Low',
                                                 self.ids_low,
                                                 self.data_dir,
                                                 self.lstm_time_spread, 
                                                samples_per_type,
                                                n_classes,
                                                self.shuffle,
                                                self.conv_w_size,
                                                self.no_input_channels,
                                                self.shot_dfs,
                                                self.no_classes,
                                                self.stride,
                                                self.conv_w_offset)
        
        self.sub_generators = [self.low_generator, self.high_generator, self.dither_generator, self.dither_generator]
        
        
        
        
        
        
    def initialize(self,shot_ids, batch_size, n_classes, shuffle,lstm_time_spread, epoch_size,
                        train_data_name, conv_w_size, no_input_channels,gaussian_hinterval,
                        no_classes, stride, labelers,conv_w_offset, fixed_inds, signal_sampling_rate, machine_id='tcv', normalize_per_shot=True,
                        data_dir = './labeled_data/TCV'):
        # print(shot_ids)
        # exit(0)
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        self.lstm_time_spread = int(lstm_time_spread)
        self.ids_trans = IDsAndLabelsLSTM()
        self.ids_non_trans = IDsAndLabelsLSTM()
        
        self.ids_none = IDsAndLabelsLSTM()
        self.ids_low = IDsAndLabelsLSTM()
        self.ids_dither = IDsAndLabelsLSTM()
        self.ids_high = IDsAndLabelsLSTM()
        self.ids_lowstarts = IDsAndLabelsLSTM()
        # self.ids = [self.ids_DMode, self.ids_HMode, self.ids_LMode]
        # self.ids = [self.ids_trans, self.ids_non_trans]
        self.ids = [self.ids_none, self.ids_low, self.ids_dither, self.ids_high]
        self.shot_ids = ()
        self.no_labelers = len(labelers)
        self.labelers = labelers
        # print(shot_ids, labelers)
        self.intc_times = {}
        self.non_consensus_times= { }
        self.normalize_per_shot = normalize_per_shot
        for s in shot_ids:
            for l in self.labelers:
                self.shot_ids += (str(s) + '-' + str(l),)
                # print(s, l)
                # exit(0)
            #for each shot, get times where there is no consensus
            # try:
            intc_times = load_shot_and_equalize_times(self.data_dir, s, self.labelers, signal_sampling_rate)
            self.intc_times[str(s)] = intc_times
            # except:
            #     print('could not load intersect times for shot', s, ',please check for errors.')
            #     exit(0)
            # print('finished')
            # exit(0)
            # labeler_states, labeler_elms = get_different_labels(self.data_dir, s, self.labelers, intc_times)
            # mode_labeler_states = calc_consensus(labeler_states.swapaxes(0,1))
           
        # exit(0)
        # print('hrthrth')    
        self.shuffle = shuffle
        self.length = epoch_size
        self.shot_dfs = {}
        self.conv_w_size = conv_w_size
        self.no_input_channels = no_input_channels
        self.no_classes = no_classes
        self.full_shot_dfs = {}
        rep_checker = {}
        self.shot_block_ids = {}
        self.shot_total_ids = {}
        self.stride = stride
        time_labeler_track = {0:[], 1:[]}
        labeler_counter = -1
        # itsc_times = {}
        self.shuffle = shuffle
        self.conv_w_offset = conv_w_offset
        self.fixed_inds = fixed_inds
        
        # 
        # self.ids_non_trans = IDsAndLabelsCNN()
        # self.ids_lh_trans = IDsAndLabelsCNN()
        # self.ids_hl_trans = IDsAndLabelsCNN()
        # self.ids_hd_trans = IDsAndLabelsCNN()
        # self.ids_dh_trans = IDsAndLabelsCNN()
        # self.ids_ld_trans = IDsAndLabelsCNN()
        # self.ids_dl_trans = IDsAndLabelsCNN()
        # self.ids_elms = IDsAndLabelsCNN()
        # self.ids_dithers = IDsAndLabelsCNN()
        self.gaussian_hinterval = gaussian_hinterval
        self.machine_id = machine_id
        
        
    def first_prepro_cycle(self,):
        gaussian_hinterval = self.gaussian_hinterval
        for shot in self.shot_ids:
            fshot, fshot_times = load_fshot_from_labeler(shot, self.machine_id, self.data_dir)
            fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
            fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values_gauss(fshot.LHD_label.values,
                                                                                                                                            fshot.time.values,
                                                                                                                                            smooth_window_hsize=gaussian_hinterval)
            #CUTOFF to put all elm labels at 0 where state is not high
            fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0
            
            fshot = state_to_trans_event_disc(fshot, gaussian_hinterval)
            fshot = trans_disc_to_cont(fshot, gaussian_hinterval)
        
            self.shot_dfs[str(shot)] = fshot.copy()
    
    def sec_prepro_cycle(self,):
        labeler_counter = -1
        ss = []
        itsc_times = self.intc_times
        # print([type(k) for k in itsc_times.keys()])
        shots_id_list = self.shot_dfs.keys()
        self.norm_factors = {}
        for shot in shots_id_list:
            # print('in second cycle')
            shot_no = shot[:5]
            # print(shot)
            labeler_counter += 1
            labeler_track = labeler_counter % self.no_labelers
            # time_labeler_track[labeler_track] = fshot_times
            labeler = self.labelers[labeler_track]
            # self.shot_dfs[str(shot)] = fshot.copy()
            labeler_intersect_times = itsc_times[shot_no]
            # print(len(labeler_intersect_times))
            fshot = self.shot_dfs[str(shot)].copy()
            fshot = fshot[fshot['time'].round(5).isin(labeler_intersect_times)]
            if self.normalize_per_shot:
                fshot = normalize_signals_mean(fshot) #NORMALIZATION CAN ONLY HAPPEN AFTER SHOT FROM BOTH LABELERS HAS BEEN ASSERTED TO BE THE SAME!
            
            # df[df['time'] in labeler_intersect_times]
            self.shot_dfs[str(shot)] = fshot
            
            ss += [np.round(fshot.PD.values, 10)]
            # print(len(ss))
            if labeler_track == 0:
                # fig = plt.figure()
                pass
            if labeler_track == self.no_labelers - 1: #compare the photodiode values of the shots coming from all labelers with each other. They should be exactly the same!
                # print('checking!')
                for s_id in range(1, len(ss)):
                    # print(ss[s_id], len(ss[s_id]))
                    # print(ss[s_id - 1], len(ss[s_id - 1]))
                    # print(ss[s_id], len(ss[s_id]))
                    # print(ss[s_id - 1], len(ss[s_id - 1]))
                    # plt.plot(ss[s_id - 1])
                    # plt.plot(ss[s_id])
                    # plt.show()
                    # if(np.array_equal(ss[s_id], ss[s_id - 1]) == False):
                    #     print('problem in this shot', shot)
                    pass 
                ss = []
                # plt.plot(fshot['time'].values, fshot['PD'].values, label=labeler)
                # plt.plot(fshot['time'].values, fshot['sm_low_label'].values, label=labeler+'lhd')
                # plt.legend()
                # fig.suptitle(shot[:5])
                # plt.show()
                
            else:
                # print('checking else!')
                # plt.xlabel('t (s)')
                # plt.ylabel('PD')
                # plt.plot(fshot['time'].values, fshot['PD'].values, label=labeler)
                # plt.plot(fshot['time'].values, fshot['sm_low_label'].values, label=labeler+'lhd')
                pass
        if not self.normalize_per_shot:
            print('not normalizing per shot. check LSTM data generator class, method sec_prepro_cycle')
            norm_factors = normalize_signals_whole_set(self.shot_dfs.values())
            # print(norm_factors)
            self.norm_factors = norm_factors
            
    def third_prepro_cycle(self,):
        for shot in self.shot_ids:
            fshot = self.shot_dfs[str(shot)]
            for k in range(len(fshot)):
                dt = fshot.iloc[k]
                elm_lab_in_dt = get_elm_label_in_dt(dt)
                state_lab_in_dt = get_state_labels_in_dt(dt)
                # if state_lab_in_dt[0] == 1:
                #     self.ids_none.add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)
                if state_lab_in_dt[1] > .99:
                    self.ids_low.add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)  
                elif state_lab_in_dt[2] >.99:
                    self.ids_dither.add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)
                # elif state_lab_in_dt[3] == 1:
                # print(elm_lab_in_dt)
                elif elm_lab_in_dt[0] != 0:
                    self.ids_high.add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)
                    
            
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length
    
    def __getitem__(self, index):
        
        # print 'Generate one batch of data', len(self.sub_generators)
        batch_X_scalars, batch_y_states, batch_y_elms, batch_shots_ids, batch_ts, batch_X_start, batch_y_trans = [], [], [], [], [], [], []
        for sub_generator in self.sub_generators:
            # print('fixed_inds', self.fixed_inds)
            # exit()
            X_scalars, y_states, y_elms, shots_ids, ts, X_start, y_trans = sub_generator[[index, self.fixed_inds]]
            for inputs_scalars, target_states, target_elms, shots_ids_temp, ts_temp, inputs_start, outputs_trans in zip(X_scalars, y_states, y_elms, shots_ids, ts, X_start, y_trans):
                batch_X_scalars += [inputs_scalars]
                batch_y_states += [target_states]
                batch_y_elms += [target_elms]
                batch_shots_ids += [shots_ids_temp]
                batch_ts += [ts_temp]
                batch_X_start += [inputs_start]
                batch_y_trans += [outputs_trans]
                
        batch_X_scalars = np.asarray(batch_X_scalars)
        batch_y_states = np.asarray(batch_y_states)
        batch_y_elms = np.asarray(batch_y_elms)
        batch_shots_ids = np.asarray(batch_shots_ids)
        batch_ts = np.asarray(batch_ts)
        batch_X_start = np.asarray(batch_X_start)
        batch_y_trans = np.asarray(batch_y_trans)
        
        aux = list(zip(
                        np.asarray(batch_X_scalars),
                        np.asarray(batch_y_states),
                        np.asarray(batch_y_elms),
                        np.asarray(batch_shots_ids),
                        np.asarray(batch_ts),
                        np.asarray(batch_X_start),
                        np.asarray(batch_y_trans)
                        ))
        random.shuffle(aux)
        batch_X_scalars, batch_y_states, batch_y_elms, batch_shots_ids, batch_ts, batch_X_start, batch_y_trans = zip(*aux)

        return (
                {
                    'in_scalars':np.asarray(batch_X_scalars),
                    'in_seq_start': np.asarray(batch_X_start)
                },
                {
                    'out_states':np.asarray(batch_y_states),
                    'out_elms': np.asarray(batch_y_elms),
                    'out_transitions': np.asarray(batch_y_trans)
                },
                {
                    'times': np.asarray(batch_ts),
                    'shots_ids': np.asarray(batch_shots_ids),
                    
                }
                )
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
            # for sub_generator in self.sub_generators:
            #         sub_generator.on_epoch_end()
            # self.on_epoch_end()
            if self.shuffle == True:
                # print('Generator epoch finished, reshuffling...')
                self.on_epoch_end()
            
            
    def on_epoch_end(self):
        print('\n Generator epoch finished.')
        # for sid in self.shot_ids:
        for generator in self.sub_generators:
            generator.on_epoch_end()
            
    
    def compress_momentum(self, shot, prob_threshold=.005, stochastic = False):
        def trans_occur(dt):
            return (dt.sm_low_label > 0 and dt.sm_low_label < 1) or (dt.sm_dither_label > 0 and dt.sm_dither_label < 1) or (dt.sm_high_label > 0 and dt.sm_high_label < 1)
        compressed_shot = pd.DataFrame(columns=shot.columns)
        skip = 1
        skip_size = 2
        gamma = 1
        comp_factor = 21
        loc = 0
        # print('did with threshold')
        while loc < len(shot) - 1:
            # print(loc,)
            skip -= 1
            dt = shot.iloc[loc]
            next_dt = shot.iloc[loc+1]
            # conv_dt = cnn_out_trans[loc]
            # # print(conv_dt)
            # conv_elm_dt = cnn_out_elms[loc]
            #event = yes:
            if trans_occur(next_dt):# or dt.sm_elm_label != 0:
                skip_size = comp_factor
                compressed_shot = compressed_shot.append(shot.iloc[loc:loc+comp_factor].copy())
                skip = skip_size
                skip_size += gamma * skip_size
                loc += comp_factor - 1
            else:
                # rand = np.random.uniform()
                if skip == 0: #or rand < prob_threshold:
                    compressed_shot = compressed_shot.append(shot.iloc[loc:loc+comp_factor].copy())
                    skip = skip_size
                    skip_size += gamma * skip_size
                    loc += comp_factor - 1
                loc += 1
        return compressed_shot.copy() #np.asarray(compressed_cnn_out_trans), np.asarray(compressed_cnn_out_elms)
    
    def compress_random(self, shot, shot_id, prob_threshold=.005, stochastic = False):
        # print(self.shot_block_ids[shot_id])
        block_ids, tot_block_len = dict(self.shot_block_ids[shot_id]), self.shot_total_ids[shot_id]
        compressed_shot = shot.copy().reset_index()
        tot_ev_len = len(shot) - tot_block_len
        while(tot_ev_len + tot_block_len > self.lstm_time_spread + 100):
            keylist = list(block_ids.keys())
            np.random.shuffle(keylist)
            key = random.choice(keylist)
            vals = block_ids[key]
            tot_block_len -= len(vals)
            del block_ids[key]
            compressed_shot = compressed_shot.drop(vals)
        compressed_shot = compressed_shot[:(self.lstm_time_spread + self.conv_w_size)]
        # plt.plot(compressed_shot['PD'].values)
        # plt.plot(compressed_shot['IP'].values)
        # plt.plot(compressed_shot['FIR'].values)
        # plt.plot(compressed_shot['DML'].values)
        # plt.plot(compressed_shot['sm_none_label'].values)
        # plt.plot(compressed_shot['sm_low_label'].values)
        # plt.plot(compressed_shot['sm_dither_label'].values)
        # plt.plot(compressed_shot['sm_high_label'].values)
        # plt.plot(compressed_shot['sm_elm_label'].values)
        # plt.legend(['PD', 'IP', 'FIR', 'DML', 'none', 'low', 'dither', 'high', 'elm'])
        # plt.title(shot)
        # plt.show()
        # # exit(0)
        return compressed_shot
                
    def get_blocks_for_compression(self, shot):
        ids = np.arange(len(shot))
        ids_non_trans = []
        ids_trans = []
        block_ids = {}
        tot_block_len = 0
        counter = 0
        i = 0
        block_ids[i] = []
        trans = False
        for k in range(len(shot)):
            dt = shot.iloc[k]
            elm_lab_in_dt = get_elm_label_in_dt(dt)
            state_lab_in_dt = get_state_labels_in_dt(dt)
            # print(state_lab_in_dt)
            if (not np.any(np.logical_and(state_lab_in_dt < 1, state_lab_in_dt > 0))): # and state_lab_in_dt[2] == 0 :
                counter += 1
                if counter >= 0:
                    if counter%20 == 0:
                        i += 1
                        block_ids[i] = []
                        counter = 0
                    ids_non_trans += [k]
                    block_ids[i] += [k]
                    tot_block_len += 1
            else:
                ids_trans += [k]
                counter = -21
                trans = True
        return block_ids, tot_block_len
        
 
            
class LSTMRandomDataFetcherEndtoEndWOffset():
    def __init__(self, data_to_fetch, IDs_and_labels, data_dir,
                 lstm_time_spread=150, n_samples=16, n_classes=7, shuffle = True, conv_w_size=41, no_input_channels = 3, shot_dfs=[], no_classes=3, stride=1, conv_w_offset=20):
        print('starting', data_to_fetch)
        self.no_input_channels = int(no_input_channels) # + 1 #time
        self.n_samples = int(n_samples)
        self.IDs_and_labels = IDs_and_labels #should be a collection of objects
        # print(IDs_and_labels)
        self.list_IDs = IDs_and_labels.get_ids()
        self.n_classes = n_classes
        # self.shuffle = shuffle
        # self.w_spread = w_spread
        self.lstm_time_spread = lstm_time_spread
        self.data_dir = data_dir
        self.indexes = np.arange(self.IDs_and_labels.len)
        np.random.shuffle(self.indexes)
        # self.fshots = {}
        self.windowed_scalars={}
        self.data_to_fetch = data_to_fetch
        # print(self.data_to_fetch)
        self.conv_w_size = int(conv_w_size)
        self.shot_dfs = shot_dfs
        self.no_classes = no_classes
        # count = 0
        self.readjust_windowed_indexes()
        self.stride = stride
        self.data_to_fetch = data_to_fetch
        self.conv_w_offset = conv_w_offset
        
        
    def readjust_windowed_indexes(self,):
        # print('readjust_windowed_indexes', self.conv_w_size, self.IDs_and_labels.get_shots())
        for s in self.IDs_and_labels.get_shots():
            shot = self.shot_dfs[s]
            self.windowed_scalars[s] = np.empty((len(shot)-self.conv_w_size, self.conv_w_size, self.no_input_channels))
            for k in range(self.conv_w_size):
                disloc = shot.iloc[k : len(shot) - self.conv_w_size + k]
                # print(k, disloc.shape)
                self.windowed_scalars[s][:, k, 0] = disloc.FIR.values
                self.windowed_scalars[s][:, k, 1] = disloc.DML.values
                self.windowed_scalars[s][:, k, 2] = disloc.PD.values
                self.windowed_scalars[s][:, k, 3] = disloc.IP.values
        # # print(self.windowed_scalars.keys())
    
    def data_generation(self, list_IDs_temp):
        spread = self.lstm_time_spread #*2+1
        # print(self.n_samples, spread, self.conv_w_size, self.no_input_channels)
        X_scalars_windowed = np.empty((self.n_samples, spread//self.stride, self.conv_w_size, self.no_input_channels))
        X_scalars = np.empty((self.n_samples, spread//self.stride, 1))
        X_start = np.empty((self.n_samples, 3))
        y_states = np.empty((self.n_samples, spread//self.stride, 3), dtype=float)
        y_elms = np.empty((self.n_samples, spread//self.stride, 2), dtype=float)
        ts = np.empty((self.n_samples, spread//self.stride, 1), dtype=float)
        y_transitions = np.empty((self.n_samples, spread//self.stride, 7), dtype=float)
        # Generate data
        # print(list_IDs_temp)
        # print('in data gen')
        # shot_and_id = np.empty((self.n_samples, 3), dtype=str)
        shot_and_id = []
        for i, ID in enumerate(list_IDs_temp):
            shot, index = self.IDs_and_labels.get_shot_and_id(ID)
            # print(shot, index)
            # print(self.shot_dfs.keys())
            # index = 15575
            #For transition outputs
            # offset = np.random.randint(-self.lstm_time_spread//4,self.lstm_time_spread//4)
            # s_ind = index*1 + offset - self.lstm_time_spread//2
            # e_ind = index*1 + offset + self.lstm_time_spread//2
            
            #For state outputs
            offset = 0
            s_ind = index*1 + offset
            e_ind = index*1 + offset + self.lstm_time_spread
            
            min_id = self.conv_w_size + self.conv_w_offset
            # print(s_ind, min_id)
            if s_ind < min_id:
                # print(index, s_ind, e_ind)
                temp = min_id - s_ind
                # print('temp', temp)
                s_ind += temp
                e_ind += temp
            max_id = len(self.shot_dfs[shot]) - self.conv_w_size - self.conv_w_offset -1 
            # print('max_id', max_id)
            if e_ind > max_id: 
                temp = e_ind - max_id
                e_ind -= temp
                s_ind -= temp
            assert s_ind >= min_id
            assert e_ind <= max_id
            # print('s_ind', s_ind, 'e_ind', e_ind)
            # conv_indexes = np.arange(s_ind-self.conv_w_size//2, e_ind-self.conv_w_size//2, self.stride)
            # conv_indexes = np.arange(s_ind-self.conv_w_size, e_ind-self.conv_w_size, self.stride) 
            conv_indexes = np.arange(s_ind-self.conv_w_size + self.conv_w_offset, e_ind-self.conv_w_size + self.conv_w_offset, self.stride)
            lstm_indexes = np.arange(s_ind, e_ind, self.stride) #Set stride to 1 if we want full state outputs while preserving strided CNN inputs 
            # print(s_ind, e_ind, indexes, self.windowed_scalars[shot].shape)
            # scalars_windowed, scalars, states, elms, times = self.fetch_data_endtoend(shot, s_ind, e_ind) # forget about elms and transitions
            scalars_windowed, scalars, states, elms, times, transitions = self.fetch_data_endtoend(shot, conv_indexes, lstm_indexes) # forget about elms and transitions
            # print()
            # print(np.mean(scalars), np.isnan(np.mean(scalars)) )
            # print(s_ind, e_ind, scalars.shape)
            if np.any(np.isnan(scalars_windowed)):
                print('nan, this should happen', shot, index, s_ind, e_ind)
            if np.any(np.isnan(states)):
                print('nan in states, this should happen', shot, index, s_ind, e_ind)
            #     exit(0)
            
            X_scalars_windowed[i,] = np.asarray([scalars_windowed])
            y_states[i,] = states #keras.utils.to_categorical(state, num_classes=3)
            y_elms[i,] = elms
            # print(shot, s_ind, e_ind)
            # print([str(shot), str(s_ind), str(e_ind)])
            # print(np.asarray([str(shot), str(s_ind), str(e_ind)]))
            # shot_and_id[i,] = np.asarray([str(shot), str(s_ind), str(e_ind)])
            shot_and_id.append([str(shot), str(s_ind), str(e_ind)])
            # print('sidi', shot_and_id[i,])
            ts[i,] = times
            X_start[i,] = states[0]
            y_transitions[i,] = transitions
        return X_scalars_windowed, y_states, y_elms, shot_and_id, ts, X_start, y_transitions

    def fetch_data_endtoend(self, shot, conv_indexes, lstm_indexes):
        fshot = self.shot_dfs[shot]
        
        lstm_time_window = get_dt_and_time_window_windexes(lstm_indexes, fshot)
        # lstm_time_window = get_dt_and_time_window_wstartindex(start_index, end_index, fshot)
        # lstm_time_window = normalize_signals_mean(lstm_time_window)
        try:
            states = get_states_in_window(lstm_time_window).swapaxes(0, 1)
        except:
            # print(lstm_time_window.columns)
            print('error in', shot, lstm_indexes)
            print(lstm_time_window.sm_low_label.values)
            checksum = lstm_time_window[['sm_low_label', 'sm_dither_label', 'sm_high_label']].sum(axis=1)
            print(np.where(checksum != 1))
            print(lstm_time_window.ix[np.where(checksum != 1)[0]])
            # print(np.where(lstm_time_window['sm_low_label', 'sm_dither_label', 'sm_high_label'].sum(axis=1)))
            exit(0)
        #[:, np.newaxis,1]
        # temp = np.asarray(np.ones(len(states)) - states[:,1])
        # temp2 = states[:,1]
        # states = np.vstack((temp,temp2)).swapaxes(0,1)
        transitions = get_transitions_in_window(lstm_time_window)
        elms = get_elms_in_window(lstm_time_window).swapaxes(0, 1)
        scalars_windowed = self.windowed_scalars[shot][conv_indexes, :, :]
        scalars = get_IP_in_window(lstm_time_window).swapaxes(0, 1)
        times = np.expand_dims(get_times_in_window(lstm_time_window), axis=0).swapaxes(0, 1)

        return scalars_windowed, scalars, states, elms, times, transitions
    
    def __getitem__(self, obj):
        index = obj[0]
        fixed_inds = obj[1]
        if fixed_inds != None:
            print('Warning: fetching data with fixed index. I hope this is because you are testing something and not actually training a network.')
            # list_IDs_temp = [str(fixed_ind) for k in range(self.n_samples)]
            # print( type(fixed_inds), fixed_inds[0], fixed_inds[1])
            # exit(0)
            list_IDs_temp = []
            k = 0
            l = len(fixed_inds)
            while k != self.n_samples:
                list_IDs_temp.append(fixed_inds[k % l])
                k += 1
            # print('list_IDs_temp', len(list_IDs_temp), list_IDs_temp)
            # print(self.list_IDs, len(self.list_IDs))
            for val in list_IDs_temp:
                if val not in self.list_IDs:
                    print('The fixed value', val, ' you\'re looking for is not in the data.')
                    # exit(0)
            return self.data_generation(list_IDs_temp)
        
        #we require modulo division because the total amount of (actual, real) samples of transitions is much lower than that of non-trans
        #therefore, we will run out of transitions much faster, and hence its indexes property will be cycled through quickly
        #thus, we will have, on a given epoch, to cycle through the same event several times. However, since we augment each event a certain amount of times,
        #most of the data seen by the nn regarding this event will be augmented, instead of just thousands of passes through the same point. 
        # print(self.indexes, self.IDs_and_labels.len)     
        indexes = self.indexes[int((index*self.n_samples)%len(self.indexes)):int(((index+1)*self.n_samples)%len(self.indexes))]
        # Find list of IDs
        
        if len(indexes) == 0:
            print('looped through this type of data')
            ids1 = self.indexes[(index*self.n_samples)%len(self.indexes):]
            ids2 = self.indexes[:(((index+1)*self.n_samples)%len(self.indexes))]
            indexes = np.concatenate([ids1, ids2])
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.data_generation(list_IDs_temp)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.IDs_and_labels))
        np.random.shuffle(self.indexes)
        
def main():
    stateful = False
    compress = True
    randomized_compression = False
    gaussian_time_window = 5e-4
    signal_sampling_rate = 1e4
    stride = int(1)
    conv_w_size =40
    lstm_time_spread = int(1000)
    labelers = ['labit', 'ffelici', 'maurizio']
    labelers=['apau_and_marceca']
    conv_w_offset = 10
    shuffle=True
    params_lstm_random = {
              'batch_size': 16*4,
              'n_classes': 7,
              'lstm_time_spread': lstm_time_spread,
              'epoch_size': 64,
              'train_data_name': 'endtoendrandomizedshot',
            'no_input_channels' : 4,
            'conv_w_size':conv_w_size,
            'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
            'stride':stride,
            'labelers':labelers,
            'shuffle':shuffle,
            'conv_w_offset':conv_w_offset,
            # 'fixed_inds' : ["30262-ffelici/16020"],
            # 'fixed_inds' : ["58460-ffelici/15200"]
            # 'fixed_inds' : ["32195-ffelici/100"],
            # 'fixed_inds' : ["33446-ffelici/100"],
            # 'fixed_inds' : ["31650-ffelici/8050"]
            # 'fixed_inds' : ["32195-ffelici/2350"]
            # 'fixed_inds' : ["31211-ffelici/2100"]
            'fixed_inds' : ["61057-apau_and_marceca/8100"]
            }
    # # shot_ids = (61400, 39872,)# 49330, 48656, 26383,29200, 45106, 26389,)#49330,48656,)# 26383,29200, 45106, 26389, 29005, 29196, 47007, 45104, 29562,45103, 48827, 26384)
    # # shot_ids = (26386, 29511, 30043, 30044, 30197, 30225, 30262, 30268, 30290, 30310, 31211, 31554, 31650, 32592,32716,26383, 31718, 31807, 32191, 32195, 
    # #            32794, 32911 ) # 30302
    # shot_ids = (57103,26386,33459,43454,34010,32716,32191,61021,
    #             30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
    #             31650,31554,42514,39872,26383,48580,62744,32794,30310,31211,31807,
    #             47962,57751,31718,58460,57218,33188,56662,33271,30290,
    #             33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,
    #             42197,30262,42062,45103,33446,33567) # 34310 34318 58285 61053 33267 33282 61057
    # all_shots = (61057,57103,26386,33459,43454,34010,32716,32191,61021,
    #             30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
    #             31650,31554,42514,26383,48580,62744,32794,30310,31211,31807,
    #             47962,57751,31718,58460,57218,33188,56662,33271,30290,
    #             33281,30225,58182,32592, 30044,30043,29511,33942,45105,52302,42197,30262,42062,45103,33446,33567) #39872
    # all_shots = (61057,57103,26386,33459)
    # # shot_ids=(30275,)
    # # shot_ids=(34010, 32716, 32191, 32195, 32911, 32794, 30310, 31211, 47962, 30225, 58182, 32592)
    # # all_shots=(31211,)
    # # all_shots=(30262,32195)
    # # shot_ids=(33459,)
    # # all_shots=(30262,31211,33942,30290,32191,33446,30225,30268,30043,26386,31718,48580,31650,33638,26383,30197,30044,31807,61021,42514)
    # # all_shots=(32716, 32191, 32195, 32911, 32794, 31211, 47962, 58182, 32592, 30262, 57218, 34309, 33942, 33567, 33188, 59825, 48580, 42197, 56662, 34010, 58460, 30310, 33271)
    # # all_shots=(32716,32191,32195,32911,32794,31211,47962,58182,32592,30262,57218,34309,33942,33567,33188,59825,48580,42197,56662,34010,58460,30310,33271,33638,30043,52302,62744,53601)
    # # # all_shots=(61057,33281,26383,57103,30225,26386,42514,57751,31650,33446,45103,45105,33459,30268,43454,31807,39872,60097,31554,29511,42062,30290,30044,61021,31839,31718,60275,30197)
    # # all_shots=(32716,32191,32195,32911,32794,31211,47962,58182,32592,30262,57218,34309,33942,33567,33188,59825,48580,42197,56662,34010,58460,30310,33271,33638,30043,52302,62744,53601,26383,57103,30225,26386,42514,57751,30268,31807,31554,29511,30290,30044,61021,31839,31718,60275)
    # # all_shots=(32716,32191,32195,32911,32794,31211,47962,58182,32592,30262,57218,34309,33942,33567,33188,59825,48580,42197,56662,34010,58460,30310,33271,33638,30043,52302,62744,53601,26383,57103,30225,26386,42514,57751,30268,31807,31554,29511,30290,30044,61021,31839,31718,60275,61057,33281,60097,33446,45103,33459,31650,45105,30197,43454,42062)
    # # all_shots=(32716,32191,32195,32911,32794,31211,47962,58182,32592,30262,57218,34309,33942,33567,33188,59825,48580,42197,56662,34010,58460,30310,33271,33638,30043,52302,62744,53601,26383,57103,30225,26386,42514,57751,30268,31807,31554,29511,30290,30044,61021,31839,31718,60275,61057,33281,60097,33446,45103,33459,31650,45105,30197,43454,39872)
    # # all_shots=(26383,57103,30225,26386,42514,57751,30268,31807,31554,29511,30290,30044,61021,31839,31718,60275)
    # # all_shots=(61021,)
    # # all_shots=(42062,)
    # # all_shots=(32716, 32191, 32195, 32911, 32794, 31211, 47962, 58182, 32592, 30262)
    # # shot_ids=(47962,)
    # # all_shots=(34309,)
    # # all_shots=(61057,)# 33459, 32716, 61021, 60097, 60275, 32911, 30268, 33638, 31554, 42514, 32794, 31211, 31807, 47962, 31718, 33188, 56662, 30290, 30225, 33942, 33446)
    # # shot_ids = (61057, 61053)
    # # shot_ids = (33459, 43454)
    # 
    
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
    
    
    
    
    training_generator = LSTMDataGenerator(shot_ids=val_shots, **params_lstm_random)         
    # training_generator.on_epoch_end()
    print(training_generator.length)
    gen = next(iter(training_generator))
    a = []
    b = []
    c = []
    d = []
    e =[]
    f = []
    g =[]
    
    counter = 0
    for batch in gen:
        print(type(batch))
        inputs = batch[0]
        targets = batch[1]
        t = targets['out_states']
        el = targets['out_elms']
        means = np.mean(t, axis=(0,1))
        means = np.mean(el, axis=(0,1))
        ts = batch[2]
        print(type(batch), len(batch), inputs['in_scalars'].shape, targets['out_states'].shape, targets['out_elms'].shape,)
        for sample in range(params_lstm_random['batch_size']):
            # print inputs['in_transitions'][sample].shape, inputs['in_dithers'][sample].shape, inputs['in_elms'][sample].shape, targets[sample].shape, counter
            counter += 1
            # print(counter, sample)
            # print inputs['in_transitions'][sample][140:160]
            # print inputs['in_dithers'][sample][140:160]
            # print inputs['in_elms'][sample][140:160]
            # print(np.mean(inputs['in_scalars'][sample]))
            # print(np.mean(targets['out_states'][sample]))
            # print(targets['out_states'][sample].shape)
            # plt.plot(targets['out_states'][sample])
            # plt.show()
            # print('inputs', inputs['in_transitions'][sample].shape)
            a += [np.asarray(ts['shots_ids'][sample])]
            b += [np.asarray(targets['out_states'][sample])]
            c += [np.asarray(inputs['in_scalars'][sample])]
            d += [np.asarray(ts['times'][sample])]
            e += [np.asarray(inputs['in_seq_start'][sample])]
            f += [np.asarray(targets['out_elms'][sample])]
            g += [np.asarray(targets['out_transitions'][sample])]
            # if targets[sample] == np.asarray():
            #     break
            # 
        # training_generator.on_epoch_end()
        if counter == 4*params_lstm_random['batch_size']:
            break
        if np.any(inputs['in_scalars'] == float('nan')):
            break
    a = np.asarray(a)
    # print(a.shape)
    # exit(0)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    f=np.asarray(f)
    g = np.asarray(g)
    print('a', a.shape)
    print('b', b.shape)
    vals=(np.swapaxes(c,1,2))[0,conv_w_size//2:conv_w_size//2+stride,:,2].flatten('F')
    print('c', c.shape)
    # plt.plot(vals)
    # plt.show()
    print('d', d.shape)
    print('f', f.shape)
    # exit(0)
    
    
    
    
    # font = {'family' : 'normal',
    #     # 'weight' : 'bold',
    #     'size'   : 22}
    # import matplotlib
    # matplotlib.rc('font', **font)
    # import matplotlib.pyplot as plt 
    for k in range(64):
        # print(c.shape, g.shape, c.shape)
        # plot_pd_trans(f[k], c[k,:,conv_w_size-conv_w_offset-1,2], d[k], g[k])
        # plot_pd_signals_states(f[k], c[k,:,conv_w_size-conv_w_offset-1,2], d[k], b[k])
        plot_all_signals_all_trans(f[k], c[k,:,conv_w_size-conv_w_offset-1,:], d[k], g[k])
        
if __name__ == '__main__':
    main()