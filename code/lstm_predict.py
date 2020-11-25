import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint #TensorBoard
from lstm_model import *
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from plot_shot_results import plot_shot, plot_shot_full, plot_shot_cnn, plot_shot_simplified
from collections import OrderedDict
import csv
# from scipy import stats
from datetime import datetime
import pickle

def main():
    exp_args = ['train', 'test']
    args = sys.argv
    model_dir = 'experiments/' + args[1]
    epoch_to_predict = args[2]
    exp_train_dic = load_dic(model_dir + '/params_data_' + exp_args[0])
    exp_test_dic =  load_dic(model_dir + '/params_data_' + exp_args[1])
    print(exp_train_dic)
    print(exp_test_dic['shot_ids'])
    assert(len(set(exp_train_dic['shot_ids']) & set(exp_test_dic['shot_ids'])) == 0) #ensure no mix between test and train shots

    num_classes = 3
    c_offset = exp_train_dic['labelers']
    conv_window_size = exp_train_dic['conv_w_size']
    conv_w_offset = exp_train_dic['conv_w_offset']
    no_input_channels = exp_train_dic['no_input_channels']
    
    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5' 
    modelJoint = model_arc(bsize=1, conv_w_size=conv_window_size, no_input_channels=no_input_channels, timesteps=None, num_classes=num_classes)
    modelJoint.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    modelJoint.load_weights(model_path)
    modelJoint.reset_states()
    
    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    
    machine_id = 'TCV'
    # exit(0)
    for exp_arg in exp_args:
        print('--------------------------------------------------------------------------------------------------' +
              str(exp_arg)+
              '--------------------------------------------------------------------------------------------------')

        exp_dic = load_dic(model_dir + '/params_data_' + exp_arg)
        labelers = exp_dic['labelers']
        c_offset = exp_dic['labelers']
        shots = [str(s) for s in exp_dic['shot_ids']]#[:1] #[:3][21:22] [:1]
        # print(shots)
        # shots = ['31718',]
        conv_window_size = exp_dic['conv_w_size']
        conv_w_offset = exp_dic['conv_w_offset']
        no_input_channels = exp_dic['no_input_channels']
        pred_args = [model_dir, epoch_to_predict, exp_arg, labelers, conv_w_offset, shots, conv_window_size, no_input_channels, gaussian_hinterval, machine_id]
        predict(pred_args, modelJoint)
    
def predict(args, modelJoint):
    # print('------------STARTING------------')
    
    exp_arg = args[2]
    shots = args[5][ :2]
    labelers = args[3]
    gaussian_hinterval = args[8]
    
    data_dir = './labeled_data/' #+ labelers[0]
    # data_dir = '../../data4/labeled/' + labeler
    machine_id = args[9]
    data_dir = './labeled_data/' + machine_id + '/'
    X_scalars_test = []
    fshots = {}
    conv_window_size = args[6]
    # num_classes = 3
    no_input_channels = args[7]
    conv_w_offset = int(args[4])
    intersect_times_d = {}
    for i, shot in zip(range(len(shots)), shots):
        print('Reading shot', shot)
        fshot = pd.read_csv(data_dir + labelers[0] + '/TCV_'  + str(shot) + '_' + labelers[0] + '_labeled.csv', encoding='utf-8')
        shot_df = fshot.copy()
        shot_df = remove_current_30kA(shot_df)
        shot_df = remove_no_state(shot_df)
        shot_df = remove_disruption_points(shot_df)
        shot_df = shot_df.reset_index(drop=True)
        shot_df = normalize_current_MA(shot_df)
        shot_df = normalize_signals_mean(shot_df)
        
        
        intersect_times = np.round(shot_df.time.values,5)
        if len(labelers) > 1:
            for k, labeler in enumerate(labelers):
                fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
                intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
        fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
        intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
        intersect_times_d[shot] = intersect_times
        
        
        
        X_scalars_single = np.empty((len(fshot_equalized)-conv_window_size, conv_window_size, no_input_channels)) # First LSTM predicted value will correspond to index 20 of the full sequence. 
        # print(X_scalars_single.shape)
        # exit(0)
        # fshots[shot] = fshot.ix[shot_df.index.values]
        for j in np.arange(len(fshot_equalized) - conv_window_size):
            vals = fshot_equalized.iloc[j : conv_window_size + j]
            scalars = np.asarray([vals.FIR, vals.DML, vals.PD, vals.IP]).swapaxes(0, 1)
            assert scalars.shape == (conv_window_size, no_input_channels)
            X_scalars_single[j] = scalars
        
        X_scalars_test += [X_scalars_single]
        print('Preprocessed shot len is', len(fshot_equalized))
    # except:
    #     print('Shot', shot, 'does not exist in the database.')
        # exit(0)
    
    model_dir = args[0]
    epoch_to_predict = args[1]
    
    print('Predicting on shot(s)...')
    pred_states = []
    pred_elms = []
    pred_transitions =[]
    k_indexes =[]
    dice_cfs = []
    conf_mats = []
    conf_mets = []
    dice_cfs_dic = {}
    k_indexes_dic = {}
    # pred_start = datetime.now()
    array_sdir = model_dir + '/epoch_' + epoch_to_predict + '/network_np_out/' + exp_arg + '/'
    if not os.path.isdir(array_sdir):
        os.makedirs(array_sdir)
    for s_ind, s in enumerate(shots):
        print('Predicting shot ' + str(shots[s_ind]))
        modelJoint.reset_states()
        # start = datetime.now()
        states, elms = modelJoint.predict(
                                np.asarray([X_scalars_test[s_ind][:, :, :4]]),
                                batch_size=1,
                                verbose=1)
        # print('took', datetime.now() - start, 'to predict.')
        # print(states)
        print('Predicted sequence length is', str(states.shape), str(elms.shape))
        np.save(array_sdir + str(s) + '_states_pred.npy', states)
        np.save(array_sdir + str(s) + '_elms_pred.npy', elms)
        
    #     # sys.stdout.flush()
    #     # pred_transitions += [transitions]
    # pred_finish = datetime.datetime.now()
    # print('total prediction time: ', pred_finish - pred_start)
    # # CONVERT TO DISCRETE LABELS and save
    # 
    # 
    # 
    # for s_ind, s in enumerate(shots):
    #     np.save(array_sdir + str(s) + '_states_pred.npy', pred_states[s_ind])
    #     np.save(array_sdir + str(s) + '_elms_pred.npy', pred_elms[s_ind])
    #     
    #     

    
if __name__ == '__main__':
    main()