import numpy as np
import keras
import pandas as pd
import abc
from window_functions import *
from label_smoothing import *
from helper_funcs import *
from os import *
# from plot_shots_and_events import get_window_plots_nn_data
import math
# # from keras.utils import *
# from shot_state import StateMachine
import sys
import random
import matplotlib.pyplot as plt
import glob

class IDsAndLabelsCNN(object):
    def __init__(self,):
        self.ids = {}
        self.len = 0
    
    def generate_id_code(self, shot, index):
        return str(str(shot)+'/'+str(index))
    
    def add_id(self, shot, k, transitions, elms):
        code = self.generate_id_code(shot, k)
        if code in self.ids.keys():
            return
        else:
            self.ids[code] = {'transitions': transitions, 'elms':elms}
            self.len += 1
            # print 'added id', self.len, len(self.ids)
    
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
    

        
class CNNDataGenerator():
    'Generates data for Keras'
    def __init__(self, shot_ids=[], batch_size=16, n_classes=7, shuffle=True, epoch_size=1, train_data_name = '',
                 conv_w_size=40, no_input_channels = 3, gaussian_hinterval=10, labelers = [], augmented=3, conv_offset=50):
        'Initialization'
        # run that script just in case 
        # files = sorted(os.listdir('../data/processed/measurements_and_events_corrected/'))
        # smooth_events(files, '../data/processed/measurements_and_events_corrected/')
        self.batch_size = batch_size
        # self.data_dir = '~/event detection/data/processed/measurements_and_events_corrected_w_state_smooth_40wide_notchecked/'
        # self.data_dir = '~/event detection/data/processed/measurements_and_events_smooth_wcontdither/'
        self.data_dir = './labeled_data/'
        self.epoch_size = epoch_size
        # self.w_spread = w_spread
        self.ids_non_trans = IDsAndLabelsCNN()
        self.ids_lh_trans = IDsAndLabelsCNN()
        self.ids_hl_trans = IDsAndLabelsCNN()
        self.ids_hd_trans = IDsAndLabelsCNN()
        self.ids_dh_trans = IDsAndLabelsCNN()
        self.ids_ld_trans = IDsAndLabelsCNN()
        self.ids_dl_trans = IDsAndLabelsCNN()
        self.ids_elms = IDsAndLabelsCNN()
        self.ids_dithers = IDsAndLabelsCNN()
        self.ids = [self.ids_non_trans, self.ids_lh_trans, self.ids_hl_trans, self.ids_hd_trans, self.ids_dh_trans, self.ids_ld_trans, self.ids_dl_trans, self.ids_elms]
        # self.trans_shots = shots['trans_shots']
        # self.elm_shots = shots['elm_shots']
        # self.dither_shots = shots['dither_shots']
        # self.non_event_shots = shots['non_event_shots']
        self.shot_dfs = {}
        self.shot_ids = ()
        self.no_labelers = len(labelers)
        self.labelers = labelers
        time_labeler_track = {0:[], 1:[]}
        labeler_counter = -1
        itsc_times = {}
        self.w_spread = conv_w_size
        self.conv_offset = conv_offset
        for s in shot_ids:
            for l in self.labelers:
                self.shot_ids += (str(s) + '-' + str(l),)
        # print('upon data correction, remember to calculate the continuous dithers, and then to smoothen the targets!!')
        labeler_counter = -1
        
        print('Pre-processing raw data, generating smooth labels, checking for consistency between different labelers')
        for shot in self.shot_ids:
            # print(shot)
            # print('here')
            labeler_counter += 1
            fshot, fshot_times = load_fshot_from_labeler(shot, self.data_dir)
            fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0
            # fshot.reset_index()
            fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
            # trans_series = get_transition_times(fshot.LHD_label.values, smooth_window_hsize=gaussian_hinterval)
            # trans_ids = ['LH', 'HL', 'DH', 'HD', 'LD', 'DL']
            # for t_id in trans_ids:
            #     binary_event_values = trans_series[t_id]
            #     fshot[t_id] = smoothen_event_values(binary_event_values, smooth_window_size=gaussian_hinterval)
            # trans_vals = fshot[trans_ids].sum(axis=1)
            # fshot['no_trans'] = pd.Series(1 - trans_vals.values)
            fshot = state_to_trans_event_disc(fshot, gaussian_hinterval)
            fshot = trans_disc_to_cont(fshot, gaussian_hinterval)
            # fshot = remove_conv_start_end(fshot)
            # fig = plt.figure()
            # plt.plot(fshot['time'].values, fshot['PD'].values)
            # # plt.xlabel('t (s)')
            # # plt.ylabel('PD, (normalized)')
            # # plt.plot(fshot['time'].values, fshot['IP'].values)
            # # plt.plot(fshot['time'].values, fshot['FIR'].values)
            # # plt.plot(fshot['time'].values, fshot['DML'].values)
            # plt.plot(fshot['time'].values, fshot['LH'].values)
            # plt.plot(fshot['time'].values, fshot['HL'].values)
            # plt.plot(fshot['time'].values, fshot['DH'].values)
            # plt.plot(fshot['time'].values, fshot['HD'].values)
            # plt.plot(fshot['time'].values, fshot['LD'].values)
            # plt.plot(fshot['time'].values, fshot['DL'].values)
            # plt.plot(fshot['time'].values, fshot['no_trans'].values)
            # plt.legend(['PD'] + trans_ids + ['no_trans'])
            # plt.title(shot)
            # plt.show()
            # # 
            # exit(0)
            labeler_track = labeler_counter % self.no_labelers
            time_labeler_track[labeler_track] = fshot_times
            labeler = self.labelers[labeler_track]
            # if labeler == 'labit':
            #     fshot = remove_conv_start_end(fshot)
                        
            # print('shot', str(shot), 'has a total of', str(len(fshot)), 'points')
            self.shot_dfs[str(shot)] = fshot.copy()
                    
            shot_no = shot[:5]
            if labeler_track == self.no_labelers - 1:
                labeler_intersect_times = time_labeler_track[0]
                for l in range(self.no_labelers - 1):
                    times1 = time_labeler_track[l + 1]
                    labeler_intersect_times = sorted(set(np.round(labeler_intersect_times,5)) & set(np.round(times1,5)))
                itsc_times[shot_no] = labeler_intersect_times
            
        
        # print(len(itsc_times))
        # print('second cycle')
        labeler_counter = -1
        ss = []
        print('Normalizing data')
        for shot in self.shot_ids:
            shot_no = shot[:5]
            print(shot)
            labeler_counter += 1
            labeler_track = labeler_counter % self.no_labelers
            # time_labeler_track[labeler_track] = fshot_times
            labeler = self.labelers[labeler_track]
            labeler_intersect_times = itsc_times[shot_no]
            fshot = self.shot_dfs[str(shot)].copy()
            fshot = fshot[fshot['time'].round(5).isin(labeler_intersect_times)]
            
            fshot = normalize_signals_mean(fshot) #NORMALIZATION CAN ONLY HAPPEN AFTER SHOT FROM BOTH LABELERS HAS BEEN ASSERTED TO BE THE SAME!
            self.shot_dfs[str(shot)] = fshot
            ss += [fshot.PD.values]
            
            # if labeler_track == self.no_labelers - 1: #compare the photodiode values of the shots coming from all labelers with each other. They should be exactly the same!
            #     for s_id in range(1, len(ss)):
            #         # print(ss[s_id], len(ss[s_id]))
            #         # print(ss[s_id - 1], len(ss[s_id - 1]))
            #         if(np.array_equal(ss[s_id], ss[s_id - 1]) == False):
            #             print('problem in this shot')
            #     plt.plot(fshot['time'].values, fshot['PD'].values, label=labeler)
            #     # plt.plot(fshot['time'].values, fshot['sm_low_label'].values, label=labeler+'lhd')
            #     plt.legend()
            #     fig.suptitle(shot[:5])
            #     plt.show()
            #     ss = []
            # else:
            #     fig = plt.figure()
            #     plt.xlabel('t (s)')
            #     plt.ylabel('PD')
            #     plt.plot(fshot['time'].values, fshot['PD'].values, label=labeler)
            #     # plt.plot(fshot['time'].values, fshot['sm_low_label'].values, label=labeler+'lhd')
            #     pass
        
        
            fshot = self.shot_dfs[str(shot)]
            for k in range(self.w_spread, len(fshot)-self.w_spread):
                dt = fshot.iloc[k]
                transitions = get_transitions_in_dt(dt)
                elm_lab_in_dt = get_elm_label_in_dt(dt)
                # dithers = get_dither_in_dt(dt)          
                if transitions[0] != 0:
                    self.ids_lh_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[1] != 0:
                    self.ids_hl_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[2] != 0:
                    self.ids_hd_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[3] != 0:
                    self.ids_dh_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[4] != 0:
                    self.ids_ld_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[5] != 0:
                    self.ids_dl_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                elif elm_lab_in_dt[0] > .01:    
                    self.ids_elms.add_id(shot, k, transitions, elm_lab_in_dt)
                elif transitions[-1] > .98:
                    self.ids_non_trans.add_id(shot, k, transitions, elm_lab_in_dt)
                    
        print(len(self.ids_dh_trans.get_ids_and_labels()))
        print(len(self.ids_hl_trans.get_ids_and_labels()))
        print(len(self.ids_hd_trans.get_ids_and_labels()))
        print(len(self.ids_dh_trans.get_ids_and_labels()))
        print(len(self.ids_ld_trans.get_ids_and_labels()))
        print(len(self.ids_dl_trans.get_ids_and_labels()))
        print(len(self.ids_non_trans.get_ids_and_labels()))
        print(len(self.ids_elms.get_ids_and_labels()))
        #         
        #         
        # for shot in self.elm_shots:
        #     fshot = self.shot_dfs[str(shot)]
        #     for k in range(self.w_spread, len(fshot)-self.w_spread):
        #         dt = fshot.iloc[k]
        #         transitions = get_transitions_in_dt(dt)
        #         elms = get_elm_in_dt(dt)
        #         dithers = get_dither_in_dt(dt)
        #         if elms[0] != 0:
        #             self.ids_elms.add_id(shot, k, transitions, elms, dithers)
        #             
        #             
        # for shot in self.non_event_shots:
        #     fshot = self.shot_dfs[str(shot)]
        #     for k in range(self.w_spread, len(fshot)-self.w_spread):
        #         dt = fshot.iloc[k]
        #         transitions = get_transitions_in_dt(dt)
        #         elms = get_elm_in_dt(dt)
        #         dithers = get_dither_in_dt(dt)
        #         if transitions[-1] > .99 and elms[-1] > .99 :
        #             # add_id(ids_non_trans, targets, generate_id_code(shot, k))
        #             self.ids_non_trans.add_id(shot, k, transitions, elms, dithers)
        #             
        # for shot in self.dither_shots:
        #     fshot = self.shot_dfs[str(shot)]
        #     for k in range(self.w_spread, len(fshot)-self.w_spread):
        #         dt = fshot.iloc[k]
        #         transitions = get_transitions_in_dt(dt)
        #         elms = get_elm_in_dt(dt)
        #         dithers = get_dither_in_dt(dt)
        #         if dithers[0] != 0:
        #             # add_id(ids_non_trans, targets, generate_id_code(shot, k))
        #             self.ids_dithers.add_id(shot, k, transitions, elms, dithers)
        
        # for e in sorted(self.ids_dithers.get_ids()):
        #     print e
        # exit(0)
        
        no_dif_types = 8
        samples_per_type = int(batch_size/no_dif_types)
        # print 'samples_per_type', samples_per_type
        #number of samples of each type will be total batch size divided by 7 (the number of possible targets)
        self.non_trans_generator = CNNRandomDataFetcher('non_trans', self.ids_non_trans, self.shot_dfs, self.w_spread, 
                                                     samples_per_type, no_input_channels, n_classes, self.conv_offset)
        #get batch_size/7 samples of lh_trans. Of those, 1/4 will be original samples, and 3/4 will be augmented.
        #this value and the batch size must be such that b_size/(1 + this) is fully divisible 
        augmented_per_sample = 1 #1 3, 7, 15
        augmented_per_sample = samples_per_type - 1
        # print int(samples_per_type/float(1+augmented_per_sample))
        # exit(0)
        self.lh_trans_generator = CNNAugmentedDataFetcher('lh_trans', self.ids_lh_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        self.hl_trans_generator = CNNAugmentedDataFetcher('hl_trans', self.ids_hl_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        self.hd_trans_generator = CNNAugmentedDataFetcher('hd_trans', self.ids_hd_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        self.dh_trans_generator = CNNAugmentedDataFetcher('dh_trans', self.ids_dh_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        self.ld_trans_generator = CNNAugmentedDataFetcher('ld_trans', self.ids_ld_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        self.dl_trans_generator = CNNAugmentedDataFetcher('dl_trans', self.ids_dl_trans, self.shot_dfs, self.w_spread,
                                                       int(samples_per_type/float(1+augmented_per_sample)), no_input_channels, n_classes,
                                                       augmented_per_sample, self.conv_offset) #3 per actual data sample
        
        self.elm_generator = CNNRandomDataFetcher('elm', self.ids_elms, self.shot_dfs, self.w_spread,
                                               samples_per_type, no_input_channels, n_classes, self.conv_offset)
        
        # self.dither_generator = CNNRandomDataFetcher('dither', self.ids_dithers, self.data_dir, self.w_spread,
        #                                        samples_per_type, dim, n_channels, n_classes, shuffle)
        
        # self.sub_generators = [self.non_trans_generator, self.lh_trans_generator]
        
        
        self.sub_generators = [self.non_trans_generator, self.lh_trans_generator, self.hl_trans_generator, self.hd_trans_generator,
                               self.dh_trans_generator, self.ld_trans_generator, self.dl_trans_generator, self.elm_generator]#, self.dither_generator]
        # self.sub_generators = [self.ld_trans_generator]
        assert len(self.sub_generators) == no_dif_types
        
        l = 0
        for i in self.ids:
            
            l += i.len
            
        self.length = l
        print('Shuffling generator.')
        self.on_epoch_end()
        # print('finished object construction. length:', self.length)
        # self.indexes = np.arange(self.len)
        # self.__getitem__()
        # while(True):
        #     yield self.__getitem__()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        # return int(np.floor(self.length / self.batch_size))
        return self.epoch_size
        # return self.len
        # return 1
    
    def __getitem__(self, index):
        # print 'Generate one batch of data', len(self.sub_generators)
        batch_X, batch_y_trans, batch_y_elms, batch_y_dithers, batch_sids = [], [], [], [], []
        for sub_generator in self.sub_generators:
            # print sub_generator
            X, y_trans, y_elms, sids = sub_generator[index] # y_dithers
            
            for inputs, targets_trans, targets_elms, sid in zip(X, y_trans, y_elms, sids): #targets_dithers
                batch_X += [inputs]
                # print(inputs.shape)
                # batch_X +=[np.expand_dims(inputs[:,1], axis=1)]
                # batch_X +=[inputs[:,(0, 2,3)]]
                # print inputs.shape
                batch_y_trans += [targets_trans]
                # print targets_trans.shape
                batch_y_elms += [targets_elms]
                batch_sids += [sid]
                # batch_y_dithers += [targets_dithers]
        return ({'conv_input': np.asarray(batch_X),
                 # 'central_conv_input': np.asarray(batch_X)[:,40:61,:]
                 # 'spatial_info': np.cos(np.linspace(-np.pi/2, np.pi/2, 101, endpoint=True)),
                 },
            {'out_transitions': np.asarray(batch_y_trans),
             'out_elms': np.asarray(batch_y_elms),
             # 'out_dithers': np.asarray(batch_y_dithers)
             },
            {'shot_ids': batch_sids
             }
            )
    

    def __iter__(self):
        return self
    
    def __next__(self):
        counter = 0
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
                # counter += 1
                # if counter == self.epoch_size:
            print('Generator epoch finished, reshuffling...')   
            self.on_epoch_end()
                
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for sub_generator in self.sub_generators:
            sub_generator.on_epoch_end()
    
class CNNDataFetcher(object):
    def __init__(self, data_to_fetch, IDs_and_labels, shot_dfs, w_spread=50, n_samples=16, n_channels=2, n_classes=7, shuffle=True, conv_offset=50):
        # self.dim = dim
        # print 'hereeeeee', self.dim
        self.n_samples = n_samples
        self.IDs_and_labels = IDs_and_labels #should be a collection of objects
        # print(self.IDs_and_labels)
        # exit(0)
        self.list_IDs = IDs_and_labels.get_ids()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.w_spread = w_spread
        # self.data_dir = data_dir
        self.indexes = np.arange(self.IDs_and_labels.len)
        # print self.IDs_and_labels.len
        # print len(self.indexes)
        # print len(self.list_IDs)
        # print len(self.list_IDs)
        # self.fshots = {}
        # self.data_to_fetch = data_to_fetch
        self.shot_dfs = shot_dfs
        self.data_to_fetch = data_to_fetch
        self.conv_offset = conv_offset
        # for s in self.IDs_and_labels.get_shots():
        #     self.fshots[s] = pd.read_csv(self.data_dir + str(s) + '.csv').copy()
        
    def __getitem__(self, index):
        #we require modulo division because the total amount of (actual, real) samples of transitions is much lower than that of non-trans
        #therefore, we will run out of transitions much faster, and hence its indexes property will be cycled through quickly
        #thus, we will have, on a given epoch, to cycle through the same event several times. However, since we augment each event a certain amount of times,
        #most of the data seen by the nn regarding this event will be augmented, instead of just thousands of passes through the same point. 
      
        indexes = self.indexes[int((index*self.n_samples)%len(self.indexes)):int(((index+1)*self.n_samples)%len(self.indexes))]
        # Find list of IDs
        if len(indexes) == 0:
            # print 'looped through this type of data'
            ids1 = self.indexes[(index*self.n_samples)%len(self.indexes):]
            ids2 = self.indexes[:(((index+1)*self.n_samples)%len(self.indexes))]
            indexes = np.concatenate([ids1, ids2])
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        for i in list_IDs_temp:
            shot, ind  = self.IDs_and_labels.get_shot_and_id(i)
            # print('fetching', shot, ind, self.data_to_fetch)
        return self.data_generation(list_IDs_temp)
    
    def on_epoch_end(self):
        # print('epoch ended, reshuffling...')
        self.indexes = np.arange(self.IDs_and_labels.len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # print 'epoch ended.'
    
    def fetch_data(self, ID):
        shot, ind = self.IDs_and_labels.get_shot_and_id(ID)
        # ind=2381
        # print('fetching', shot, ind, self.data_to_fetch)
        fshot = self.shot_dfs[str(shot)]
        # scalars_window = np.empty((self.w_spread, self.n_channels))
        # fshot['fir'] = pd.Series(fshot.fir.values/np.mean(fshot.fir.values))
        # fshot['PD'] = pd.Series(fshot.PD.values/np.mean(fshot.PD.values))
        # fshot['DML'] = pd.Series(fshot.DML.values/np.mean(fshot.DML.values))
        current_dt, window = get_dt_and_window(int(ind), self.w_spread, fshot, self.conv_offset)
        scalars_window = get_raw_signals_in_window(window)
        # print(fir_vals.shape, pd_vals.shape)
        # print(self.w_spread, scalars_window.shape)
        assert scalars_window.shape[1] == self.w_spread
        # scalars_window = np.asarray([fir_vals, dml_vals, pd_vals])
        elms = get_elm_label_in_dt(current_dt)
        transitions = get_transitions_in_dt(current_dt)
        # print(transitions)
        assert np.sum(transitions) == 1
        return scalars_window.swapaxes(0,1), transitions, elms, str(shot) + str(ind)#, np.cos(np.linspace(-np.pi/2, np.pi/2, 101, endpoint=True))


class CNNRandomDataFetcher(CNNDataFetcher):
    def __init__(self, data_to_fetch, IDs_and_labels, shot_dfs, w_spread=50, n_samples=16, n_channels=2, n_classes=7, conv_offset=50):
        CNNDataFetcher.__init__(self, data_to_fetch, IDs_and_labels, shot_dfs, w_spread=w_spread, n_samples=n_samples, n_channels=n_channels, n_classes=n_classes, conv_offset=conv_offset)
   
   
    def data_generation(self, list_IDs_temp):
        X = np.empty((self.n_samples, self.w_spread, self.n_channels))
        y_trans = np.empty((self.n_samples, self.n_classes), dtype=float)
        y_elms = np.empty((self.n_samples, 2), dtype=float)
        # Generate data
        shot_and_id =[]
        for i, ID in enumerate(list_IDs_temp):
            scalars_window, transitions, elms, sid = self.fetch_data(ID) #, spatial_inf
            X[i,] = scalars_window
            label = self.IDs_and_labels.get_label(ID)
            y_trans[i] = transitions
            y_elms[i] = elms
            shot_and_id.append([sid])
        return X, y_trans, y_elms, shot_and_id#, y_dithers
    

        
class CNNAugmentedDataFetcher(CNNDataFetcher):
    #no. augmented samples must be such that everything is divisible by 4.
    # for eaxmple, if I want 4 samples in total of a particular label, what I will call the superconstructor with is 1 (the remaining 3 will be augmented).
    # If I want 8 sampels in total, superconstructor gets nsamples = 2, the remaining 6 are augmented
    def __init__(self, data_to_fetch, IDs_and_labels, shot_dfs, w_spread=50, n_samples=16, n_channels=2, n_classes=7, augmented_samples=3,  conv_offset=50):
        CNNDataFetcher.__init__(self, data_to_fetch, IDs_and_labels, shot_dfs, w_spread=w_spread, n_samples=n_samples, n_channels=n_channels, n_classes=n_classes, conv_offset=conv_offset)
        self.n_augmented_samples = augmented_samples
    
    def augment_sample(self, real_vals):
        # real_fir_vals, real_dml_vals, real_pd_vals = real_vals[], real_vals, real_vals
        random_vals_window = np.random.random_sample()#- .5
        # random_vals_scalars = np.random.random_sample((2,)) + .5
        # augmented_fir_vals = random_vals_window[0] * real_fir_vals# / 1e19
        # augmented_pd_vals = random_vals_window[0] * real_pd_vals
        # augmented_dml_vals = random_vals_window[0] * real_dml_vals
        augmented_vals = real_vals * random_vals_window
        # return augmented_fir_vals, augmented_dml_vals, augmented_pd_vals
        return augmented_vals
    
    def data_generation(self, list_IDs_temp):
        if len(list_IDs_temp) == 0:
            return np.zeros((self.n_samples*(self.n_augmented_samples+1), self.w_spread, self.n_channels)), np.zeros((self.n_samples*(self.n_augmented_samples+1), self.n_classes), dtype=float)
        shot_and_id = []
        X = np.empty((self.n_samples*(self.n_augmented_samples+1), self.w_spread, self.n_channels))
        # print X.shape, self.n_samples, (self.n_augmented_samples+1)
        # y = np.empty((self.n_samples*(self.n_augmented_samples+1), self.n_classes), dtype=float)
        y_trans = np.empty((self.n_samples*(self.n_augmented_samples+1), self.n_classes), dtype=float)
        y_elms = np.empty((self.n_samples*(self.n_augmented_samples+1), 2), dtype=float)
        # y_dithers = np.empty((self.n_samples*(self.n_augmented_samples+1), 2), dtype=float)
        # counter = 0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            scalars_window, transitions, elms, sid = self.fetch_data(ID)
            # print fir_vals, pd_vals, X.shape
            label = self.IDs_and_labels.get_label(ID)
            X[i*(self.n_augmented_samples+1)] = scalars_window
            # y_trans[i*(self.n_augmented_samples+1)] = label
            y_trans[i*(self.n_augmented_samples+1)] = transitions
            y_elms[i*(self.n_augmented_samples+1)] = elms
            # y_dithers[i*(self.n_augmented_samples+1)] = label['dithers']
            shot_and_id.append([sid])
            for k in range(1, self.n_augmented_samples + 1):
                augmented_vals = self.augment_sample(scalars_window)
                X[i*(self.n_augmented_samples+1) + k] = augmented_vals
                y_trans[i*(self.n_augmented_samples+1) + k] = transitions
                y_elms[i*(self.n_augmented_samples+1) + k] = elms
                # y_dithers[i*(self.n_augmented_samples+1) + k] = label['dithers']
                shot_and_id.append([sid])
        return X, y_trans, y_elms, shot_and_id#, y_dithers

# 



def main():
    # print('HERE')
    stateful = False
    compress = True
    randomized_compression = False
    gaussian_time_window = 1e-3
    signal_sampling_rate = 1e4
    conv_w_size = 200
    labelers = ['labit', 'ffelici']
    # labelers = ['ffelici']
    params_random = {
                'batch_size': 8*2,
                'n_classes': 7,
                'shuffle': True,
                'epoch_size': 2,
                'train_data_name': 'endtoendrandomizedshot',
                'no_input_channels' : 4,
                'conv_w_size':conv_w_size,
                'gaussian_hinterval': int(gaussian_time_window * signal_sampling_rate),
                'labelers':labelers,
                'conv_offset':10}
    # shot_ids = (61400, 39872,)# 49330, 48656, 26383,29200, 45106, 26389,)#49330,48656,)# 26383,29200, 45106, 26389, 29005, 29196, 47007, 45104, 29562,45103, 48827, 26384)
    # shot_ids = (26386, 29511, 30043, 30044, 30197, 30225, 30262, 30268, 30290, 30310, 31211, 31554, 31650, 32592,32716,26383, 31718, 31807, 32191, 32195, 
    #            32794, 32911 ) # 30302
    shot_ids = (57103,26386,33459,43454,34010,32716,32191,61021,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
                31650,31554,42514,39872,26383,48580,62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,
                33281,30225,58182,32592,30044,30043,29511,33942,45105,52302,
                42197,30262,42062,45103,33446,33567) # 34310 34318 58285 61053 33267 33282 61057
    all_shots = (61057,57103,26386,33459,43454,34010,32716,32191,61021,
                30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
                31650,31554,42514,39872,26383,48580,62744,32794,30310,31211,31807,
                47962,57751,31718,58460,57218,33188,56662,33271,30290,
                33281,30225,58182,32592, 30044,30043,29511,33942,45105,52302,42197,30262,42062,45103,33446,33567)
    # shot_ids=(45105, 30262)
    all_shots = (34010, 32716, 32191, 32195, 32911, 32794, 30310, 31211, 47962, 30225, 58182, 32592)
    # all_shots=(34010,)
    all_shots= (30262,31211,33942,30290,32191,)#33446,30225,30268,30043,26386,31718,48580,31650,33638,26383,30197,30044,31807,61021,42514)
    # # shot_ids=(47962,)
    # all_shots=(34309,)
    # all_shots=(61057,)# 33459, 32716, 61021, 60097, 60275, 32911, 30268, 33638, 31554, 42514, 32794, 31211, 31807, 47962, 31718, 33188, 56662, 30290, 30225, 33942, 33446)
    # shot_ids = (61057, 61053)
    # shot_ids = (33459, 43454)
    
    training_generator = CNNDataGenerator(shot_ids=(all_shots), **params_random)         
    # training_generator.on_epoch_end()
    print(training_generator.length)
    gen = next(iter(training_generator))
    a = []
    b = []
    c = []
    d = []
    
    
    counter = 0
    for batch in gen:
        # print(type(batch))
        inputs = batch[0]
        targets = batch[1]
        sids = batch[2]
        # print(type(batch), len(batch), inputs['in_transitions'].shape, inputs['in_dithers'].shape, targets.shape,)
        for sample in range(params_random['batch_size']):
            # print inputs['in_transitions'][sample].shape, inputs['in_dithers'][sample].shape, inputs['in_elms'][sample].shape, targets[sample].shape, counter
            counter += 1
            # print(counter, sample)
            # print inputs['in_transitions'][sample][140:160]
            # print inputs['in_dithers'][sample][140:160]
            # print inputs['in_elms'][sample][140:160]
            
            # print('inputs', inputs['conv_input'][sample].shape)
            # print('targets', targets['out_elms'][sample].shape, targets['out_transitions'][sample].shape)
            a += [np.asarray(inputs['conv_input'][sample])]
            b += [np.asarray(targets['out_elms'][sample])]
            c += [np.asarray(targets['out_transitions'][sample])]
            d += [sids['shot_ids'][sample]]
            # c += [np.asarray(inputs['in_scalars'][sample])]
            # d += [np.asarray(inputs['in_elms'][sample])]
            # if targets[sample] == np.asarray():
            #     break
            # 
        # if counter%8 == 0:
        #     training_generator.on_epoch_end() #batch over
        if counter > params_random['batch_size']:
            break
    a = np.asarray(a)
    # print a
    b = np.asarray(b)
    c = np.asarray(c)
    # d = np.asarray(d)
    print('a', a.shape)
    print('b', b.shape)
    print('c', c.shape)
    
   
    for sample in range(params_random['batch_size']):
        # continue
        # print inputs['conv_input'][sample, 11, 0], targets['out_transitions'][sample], targets['out_elms'][sample], targets['out_dithers'][sample]
        # print inputs['wide_conv_input'].shape
        fig = plt.figure()
        # fig.suptitle(d[sample])
        p = fig.add_subplot(131)
        p.plot(np.arange(conv_w_size), a[sample, :, 0])
        p = fig.add_subplot(132)
        p.plot(np.arange(conv_w_size), a[sample, :, 1])
        p = fig.add_subplot(133)
        p.plot(np.arange(conv_w_size), a[sample, :, 2])
        # p = fig.add_subplot(143)
        # p.plot(np.arange(21), inputs['central_conv_input'][sample, :, 0])
        # p = fig.add_subplot(144)
        # p.plot(np.arange(21), inputs['central_conv_input'][sample, :, 1])
        fig.suptitle(str(c[sample]) + str(b[sample]) +str(d[sample]))
        
        plt.show()
            
 
if __name__ == '__main__':
    main()