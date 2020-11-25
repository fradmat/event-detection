import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from model import *
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from plot_shot_results import plot_shot
from collections import OrderedDict
import csv

def main():
    predict(sys.argv)
    
    
def predict(args):
    print('------------STARTING------------')
    gaussian_time_window = 1e-3
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    exp_arg = args[3]
    shots = args[4].split(",")
    data_dir = '../../data3'
    X_scalars_test = []
    fshots = {}
    conv_window_size = 40
    num_classes = 3
    no_input_channels = 3
    for i, shot in zip(range(len(shots)), shots):
        print('Reading shot', shot)
        # continue
        # try:
        # fshot = pd.read_csv(data_dir +'/test/TCV_'  + str(shot) + '_signals.csv', encoding='utf-8')
        fshot = pd.read_csv(data_dir +'/labeled/TCV_'  + str(shot) + '_ffelici_labeled.csv', encoding='utf-8')
        shot_df = fshot.copy()
        shot_df = remove_current_30kA(shot_df)
        shot_df = remove_no_state(shot_df)
        fshot = fshot.reset_index()
        # shot_df = shot_df[10:len(shot_df)-10]
        shot_df = normalize_current_MA(shot_df)
        
        # shot_df['FIR'] = shot_df['FIR'].divide(1e19)
        # shot_df = shot_df.reset_index()
        shot_df = normalize_signals_mean(shot_df)
        # plt.plot(shot_df.FIR.values)
        # plt.plot(shot_df.PD.values)
        # plt.plot(shot_df.DML.values)
        # plt.plot(shot_df.LHD_label.values)
        # plt.show()
        X_scalars_single = np.empty((len(shot_df)-conv_window_size, conv_window_size, no_input_channels)) # First LSTM predicted value will correspond to index 20 of the full sequence. 
        fshots[shot] = fshot.ix[shot_df.index.values]
        for j in np.arange(len(shot_df) - conv_window_size):
            vals = shot_df.iloc[j : conv_window_size + j]
            # scalars = np.asarray([vals.FIR, vals.DML, vals.PD, vals.IP]).swapaxes(0, 1)
            # scalars = np.asarray([vals.FIR, vals.PD]).swapaxes(0, 1)
            scalars = np.asarray([vals.FIR, vals.DML, vals.PD]).swapaxes(0, 1)
            assert scalars.shape == (conv_window_size, no_input_channels)
            X_scalars_single[j] = scalars
        X_scalars_test += [X_scalars_single]
        print('Preprocessed shot len is', len(shot_df))
    # except:
    #     print('Shot', shot, 'does not exist in the database.')
        # exit(0)
    
    model_dir = args[1]
    epoch_to_predict = args[2]
    # model_path = model_dir + '/model_weights.h5'
    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5' 
    modelJoint = model_arc(bsize=1, conv_w_size=conv_window_size, no_input_channels=no_input_channels, timesteps=None, num_classes=num_classes)
    modelJoint.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # model_path = model_dir + '/model_checkpoints/modelold.h5' 
    modelJoint.load_weights(model_path)
    modelJoint.reset_states()
    
    print('Loading from,', model_path)
    print('Predicting on shot(s)...')
    pred_states = []
    pred_elms = []
    dice_cfs = []
    conf_mats = []
    conf_mets = []
    dice_cfs_dic = {}
    for k in range(len(shots)):
        print('Predicting shot ' + str(shots[k]))
        modelJoint.reset_states()
        # plt.plot(X_scalars_test[k][:,20,0])
        # plt.plot(X_scalars_test[k][:,20,1])
        # plt.plot(X_scalars_test[k][:,20,2])
        # plt.plot(X_scalars_test[k][:,20,3])
        # plt.show()
        states, elms = modelJoint.predict(
                                np.asarray([X_scalars_test[k][:, :, :4]]),
                                batch_size=1,
                                verbose=1)
        print('Predicted sequence length is', str(states.shape), str(elms.shape))
        # print(elms[0])
        # plt.plot(elms[0])
        # plt.plot(states[0])
        # plt.show()
        pred_states += [states]
        pred_elms += [elms]
       
    # CONVERT TO DISCRETE LABELS and save
    print('Post processing, saving .csv file(s)...')
    for i, shot in enumerate(shots):
        print('----------------------------------------SHOT', str(shot), '-----------------------------------------')
        # fshot = fshots[shot].reset_index()
        fshot = pd.read_csv(data_dir +'/labeled/TCV_'  + str(shot) + '_ffelici_labeled.csv', encoding='utf-8')
        fshot = remove_current_30kA(fshot)
        fshot = remove_no_state(fshot)
        fshot = fshot.reset_index()
        fshot = fshot[20:len(fshot)-20]
        # print(fshot)
        pred_elms_disc = elms_cont_to_disc(pred_elms[i][0,:,0], threshold = .75)
        pred_elms_disc = pred_elms_disc[:len(fshot)]
        
        # print(pred_states[i])
        # print(pred_states[i].shape)
        # t1 = np.copy(pred_states[i][0,:,1])
        # t2 = np.copy(pred_states[i][0,:,2])
        # pred_states[i][0,:,1] = t2
        # pred_states[i][0,:,2] = t1
        
        pred_states_disc = np.argmax(pred_states[i][0,:], axis=1)
        pred_states_disc = pred_states_disc[:len(fshot)]
        # print(len(pred_states_disc))
        fshot_labeled = pd.read_csv(data_dir +'/labeled/TCV_'  + str(shot) + '_ffelici_labeled.csv', encoding='utf-8')
        # print(len(fshot_labeled))
        intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(fshot.time.values,5))), 5)
        fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]
        # print(len(fshot_sliced), len(pred_states_disc[:len(fshot_sliced)]), len(fshot), len(pred_states_disc))
        dice_cf = dice_coefficient(pred_states_disc[:len(fshot_sliced)], fshot_sliced['LHD_label'].values)
        conf_mat = elm_conf_matrix(pred_elms_disc[:len(fshot_sliced)], fshot_sliced['ELM_label'].values, gaussian_hinterval=10, signal_times=[])
        # conf_met = conf_metrics(conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3])
        dice_cfs += [dice_cf]
        conf_mats += [conf_mat]
        dice_cfs_dic[shot] = dice_cf
        # conf_mets += [conf_met]
        # print(fshot)
        # fshot = fshot.reset_index()
        fshot['L_prob'] = pred_states[i][0,:, 0]
        fshot['H_prob'] = pred_states[i][0,:, 2]
        fshot['D_prob'] = pred_states[i][0,:, 1]
        fshot['ELM_prob'] = pred_elms[i][0,: ,0]
        fshot['ELM_det'] = pred_elms_disc
        fshot['LHD_det'] = pred_states_disc
        
        # print(fshot_sliced.LHD_label)
        
        print(len(pred_states_disc[:len(fshot_sliced)]), len(fshot_sliced))
        print('Dice coeficients for this shot (L-D-H-total):', dice_cf)
        print('Confusion matrix for this shot (TP, FP, TN, FN):', conf_mat.astype(int))
        
        pdf_save_dir = model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + 'shot' + shot + '.pdf'
        plot_shot(shot, fshot.copy(), pdf_save_dir)
        
        
        print('TCV_'  + str(shot) + '_LSTM_det.csv')
        fshot.to_csv(columns=['time', 'IP', 'FIR', 'PD', 'DML', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'],
                          path_or_buf=data_dir + '/LSTM_predicted/' + model_dir +'/TCV_'  + str(shot) + '_LSTM_det.csv', index=False)
    for k, state in enumerate(['Low', 'Dither', 'High']):
        print('Shots ordered by lowest do highest dice, sorted by ' + state)
        dd = OrderedDict(sorted(dice_cfs_dic.items(), key=lambda x: x[1][k]))
        with open(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '_sorted_scores_' + state + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, ['shot', state + ' DC'])
            w.writeheader()
            for key, val in dd.items():
                w.writerow({'shot': key, state + ' DC': val[k]})
            
                
    dice_cfs = np.asarray(dice_cfs)
    # print(dice_cfs.shape)
    f = PdfPages(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + epoch_to_predict + 'histogram.pdf')
    fig = plt.figure()
    for k, state in enumerate(['Low', 'Dither', 'High']):
        dice_state = dice_cfs[:, k]
        print(dice_state)
        p = fig.add_subplot(1,3,k+1)
        p.hist(dice_state, bins=10, range=(0, 1))
        # fig.suptitle(state)
        p.set_title(state)
        p.set_ylim(bottom = None, top=dice_state.shape[0] + 1)
        fig.suptitle('Dice coeficient histogram')
    # plt.show()
    fig.savefig(f, format='pdf')
    f.close()
    print('mean dice cfs', np.mean(dice_cfs, axis=0))
    temp = np.sum(np.asarray(conf_mats), axis=0)
    print('sum conf mats', temp)
    # print('mean conf mets', np.mean(np.asarray(conf_mets), axis=0))
    print('mean conf mets', conf_metrics(temp[0], temp[1], temp[2], temp[3]))
    
if __name__ == '__main__':
    main()