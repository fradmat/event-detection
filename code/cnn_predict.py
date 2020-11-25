import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from cnn_model import *
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from plot_shot_results import plot_shot_cnn, plot_shot_cnn_full
from collections import OrderedDict
import csv
import datetime
import operator

def main():
    exp_args = ['train', 'test']
    args = sys.argv
    model_dir = 'experiments/' + args[1]
    epoch_to_predict = args[2]
    exp_train_dic = load_dic(model_dir + '/params_data_' + exp_args[0])
    exp_test_dic = load_dic(model_dir + '/params_data_' + exp_args[1])
    print(exp_train_dic)
    print(exp_test_dic)
    
    assert(len(set(exp_train_dic['shot_ids']) & set(exp_test_dic['shot_ids'])) == 0) #ensure no mix between test and train shots
    
    # num_classes = 3   
    c_offset = exp_train_dic['labelers']
    conv_window_size = exp_train_dic['conv_w_size']
    conv_w_offset = exp_train_dic['conv_offset']
    no_input_channels = exp_train_dic['no_input_channels']
    
    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5' 
    model = cnn(conv_w_size=conv_window_size,conv_channels=no_input_channels)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.load_weights(model_path)
    
    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    
    dics = [exp_train_dic, exp_test_dic]
    for exp_arg in exp_args:
        print('--------------------------------------------------------------------------------------------------' +
              str(exp_arg)+
              '--------------------------------------------------------------------------------------------------')
        rdir = model_dir + '/' + epoch_to_predict + '/' + exp_arg
        exp_dic = load_dic(model_dir + '/params_data_' + exp_arg)
        labelers = exp_dic['labelers']
        c_offset = exp_dic['labelers']
        shots = [str(s) for s in exp_dic['shot_ids']] #[:2]
        # shots = ['31718',]
        conv_window_size = exp_dic['conv_w_size']
        conv_w_offset = exp_dic['conv_offset']
        no_input_channels = exp_dic['no_input_channels']
        pred_args = [model_dir, epoch_to_predict, exp_arg, labelers, conv_w_offset, shots, conv_window_size, no_input_channels, gaussian_hinterval]
        # exit(0)
        predict(pred_args, model)
    
    
def predict(args, model):
    print('------------STARTING------------')
    exp_arg = args[2]
    shots = args[5]
    labelers = args[3]
    gaussian_hinterval = args[8]
    model_dir = args[0]
    data_dir = './labeled_data/' 
    X_scalars_test = []
    fshots = {}
    conv_window_size = args[6]
    no_input_channels = args[7]
    conv_w_offset = int(args[4])
    epoch_to_predict = args[1]
    X_scalars_test = []
    # fshots = {}
    intersect_times_d = {}
    for i, shot in zip(range(len(shots)), shots):
        print('Reading shot', shot)
        # continue
        # try:
        # fshot = pd.read_csv(data_dir +'/test/TCV_'  + str(shot) + '_signals.csv', encoding='utf-8')
        fshot = pd.read_csv(data_dir + labelers[0] + '/TCV_'  + str(shot) + '_' + labelers[0] + '_labeled.csv', encoding='utf-8')
        shot_df = fshot.copy()
        shot_df = remove_current_30kA(shot_df)
        shot_df = remove_no_state(shot_df)
        shot_df = remove_disruption_points(shot_df)
        shot_df = shot_df.reset_index(drop=True)
        shot_df = normalize_current_MA(shot_df)
        shot_df = normalize_signals_mean(shot_df)
        
        assert len(labelers) > 1
        intersect_times = np.round(shot_df.time.values,5)
        # print('first', intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset])
        for k, labeler in enumerate(labelers):
            fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
            fshot_labeled = remove_current_30kA(fshot_labeled)
            fshot_labeled = remove_no_state(fshot_labeled)
            fshot_labeled = remove_disruption_points(fshot_labeled)
            fshot_labeled = fshot_labeled.reset_index(drop=True)
            intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
            # print(len(intersect_times))
        # print(intersect_times)
        fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
        intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
        # print(len(intersect_times))
        intersect_times_d[shot] = intersect_times
        
        
        X_scalars_single = np.empty((len(fshot_equalized)-conv_window_size, conv_window_size, no_input_channels)) # First LSTM predicted value will correspond to index 20 of the full sequence. 
        # fshots[shot] = fshot.ix[shot_df.index.values]
        # print(fshot)
        # exit(0)
        for j in np.arange(len(fshot_equalized) - conv_window_size):
            vals = fshot_equalized.iloc[j : conv_window_size + j]
            # scalars = np.asarray([vals.FIR, vals.DML, vals.PD, vals.IP]).swapaxes(0, 1)
            # scalars = np.asarray([vals.FIR, vals.PD]).swapaxes(0, 1)
            scalars = np.asarray([vals.FIR, vals.DML, vals.PD, vals.IP]).swapaxes(0, 1)
            assert scalars.shape == (conv_window_size, no_input_channels)
            X_scalars_single[j] = scalars
        X_scalars_test += [X_scalars_single]
        print('Preprocessed shot len is', len(shot_df))
    
    # exit(0)
    # print('Loading from,', model_path)
    print('Predicting on shot(s)...')
    pred_transitions = []
    pred_elm = []
    
    conf_mats = {}
    for t in get_trans_ids():
        conf_mats[t] = [0,0,0,0]
    conf_mets = []
    pred_start = datetime.datetime.now()
    array_sdir = model_dir + '/epoch_' + epoch_to_predict + '/network_np_out/' + exp_arg + '/'
    if not os.path.isdir(array_sdir):
        os.makedirs(array_sdir)
    for s_ind, s in enumerate(shots):
        print('Predicting shot ' + str(shots[s_ind]), X_scalars_test[s_ind].shape)
        elms, transitions = model.predict(
                                {'conv_input':np.asarray(X_scalars_test[s_ind])},
                                batch_size=128,
                                verbose=1
                                )
        print('Predicted sequence length is', str(transitions.shape), str(elms.shape))
        # print(elms[0])
        print(elms.shape, transitions.shape)
        np.save(array_sdir + str(s) + '_transitions_pred.npy', transitions)
        np.save(array_sdir + str(s) + '_elms_pred.npy', elms)
    #     pred_transitions += [transitions]
    #     pred_elm += [elms]
    # pred_finish = datetime.datetime.now()
    # print('total prediction time: ', pred_finish - pred_start)
    # 
    # 
    # 
    # for s_ind, s in enumerate(shots):
    #     np.save(array_sdir + str(s) + '_transitions_pred.npy', pred_transitions[s_ind])
    #     np.save(array_sdir + str(s) + '_elms_pred.npy', pred_elm[s_ind])
    
    
if __name__ == '__main__':
    main()