import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
# from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
# from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
# from matplotlib.backends.backend_pdf import PdfPages
from plot_shot_results import plot_shot_cnn, plot_shot_cnn_full, plot_shot_simplified, plot_shot_cnn_viterbi_full
# from collections import OrderedDict
import csv
import datetime
# from cnn_train import *
# import operator
from plot_scores import plot_roc_curve, plot_kappa_histogram, out_sorted_scores

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
        # if not os.path.isdir(rdir):
        #     os.makedirs(rdir)
        exp_dic = load_dic(model_dir + '/params_data_' + exp_arg)
        labelers = exp_dic['labelers']
        c_offset = exp_dic['labelers']
        shots = [str(s) for s in exp_dic['shot_ids']] #
        # shots = ['31718',]
        conv_window_size = exp_dic['conv_w_size']
        conv_w_offset = exp_dic['conv_offset']
        no_input_channels = exp_dic['no_input_channels']
        pred_args = [model_dir, '/epoch_'+epoch_to_predict, exp_arg, labelers, conv_w_offset, shots, conv_window_size, no_input_channels, gaussian_hinterval]
        # exit(0)
        predict(pred_args)
    
    
def predict(args):
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
        
    pred_transitions = []
    # pred_elm = []
    
    conf_mats = {}
    for t in get_trans_ids():
        conf_mats[t] = [0,0,0,0]
    conf_mets = []
    pred_elms = []
    array_sdir = model_dir + epoch_to_predict + '/network_np_out/' + exp_arg + '/'
    for s_ind, s in enumerate(shots):
        pred_transitions += [np.load(array_sdir + str(s) + '_transitions_pred.npy')]
        pred_elms += [np.load(array_sdir + str(s) + '_elms_pred.npy')]
        print(s, pred_transitions[-1].shape, pred_elms[-1].shape)
        
    
    thresholds = np.arange(105, step=10)/100
    metrics_weights_dic = {t:[] for t in thresholds}
    # fshots = {}
    k_indexes = []
    concat_states_labels = []
    concat_states_pred = []
    states_pred_concat = []
    ground_truth_concat = []
    consensus_concat = []
    states_pred_concat =[]
    labeler_elms_concat = []
    threshold = .1
    concat_elms = []
    concat_elm_labels = []
    k_indexes_dic = {}
    pdf_save_dir = model_dir + '/' + epoch_to_predict + '/plots/' + exp_arg + '/'
    for i, shot in enumerate(shots):
        labeler_states = []
        labeler_transitions = {}
        labeler_elms = []
        print('----------------------------------------SHOT', str(shot), '-----------------------------------------')

        fshot = pd.read_csv(data_dir+ labelers[0] +'/TCV_'  + str(shot) + '_' + labelers[0] + '_labeled.csv', encoding='utf-8')
        fshot = remove_current_30kA(fshot)
        fshot = remove_no_state(fshot)
        fshot = remove_disruption_points(fshot)
        fshot = fshot.reset_index(drop=True)
    
        intersect_times = intersect_times_d[shot]
        # print(len(intersect_times))
        for k, labeler in enumerate(labelers):
            # print(labeler)
            fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
            fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]

            labeler_states += [fshot_sliced['LHD_label'].values]
            labeler_elms += [fshot_sliced['ELM_label'].values]
            labeler_transitions[labeler] = state_to_trans_event_disc(fshot_sliced, gaussian_hinterval)
        fshot = fshot.loc[fshot['time'].round(5).isin(intersect_times)]
        
        labeler_states = np.asarray(labeler_states)
        labeler_elms = np.asarray(labeler_elms)
        # exit(0)
        fshot['ELM_label'] = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        pred_trans = pred_transitions[i][:len(fshot_sliced)]
        print(len(pred_elms), i)
        print(pred_trans.shape)
        
        
        
        pred_elms_single = pred_elms[i][:len(fshot_sliced),0]
        concat_elms.extend(pred_elms_single)
        trans_ids = get_trans_ids() + ['no_trans']
        concat_elm_labels.extend(fshot['ELM_label'].values)
        pred_elms_disc = elms_cont_to_disc(pred_elms_single, threshold=threshold, gaussian_hinterval=gaussian_hinterval)
        
        pred_states_disc, pred_trans_cat = viterbi_search(pred_trans, pred_elms_disc)
        # print(pred_trans.shape)
        # exit()
        for t, t_id in enumerate(trans_ids):
            fshot[t_id + '_det_prob'] = pred_trans[:, t]  
            fshot[t_id + '_det'] = pred_trans_cat[:, t]
            
            
            # exit(0)
        
        assert len(pred_elms_disc) == len(pred_trans)
        # print(pred_elms_disc.shape)
        # exit(0)
        
        print(pred_states_disc.shape)
        # mask = np.where(mode_labeler_states != -1)[0]
        # pred_states_disc = det_trans_to_state(fshot)
        
        
        fshot['ELM_prob'] = pred_elms_single
        fshot['LHD_det'] = pred_states_disc
        
        # pred_states_disc = states_det
        states_pred_concat.extend(pred_states_disc)
        # print(labeler_states.shape, pred_states_disc.shape)
        assert(labeler_states.shape[1] == pred_states_disc.shape[0])
        
        ground_truth = calc_mode(labeler_states.swapaxes(0,1))
        # ground_truth_elms = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        dice_cf = dice_coefficient(pred_states_disc, ground_truth)
        k_st = k_statistic(pred_states_disc, ground_truth)
        k_indexes += [k_st]
        k_indexes_dic[shot] = k_st
        
        print('calculating with majority and consensual opinion (ground truth)') #has -1 in locations where nobody agrees (case 1)
        print(len(ground_truth), sum(ground_truth == -1))
        ground_truth_concat.extend(ground_truth)
        labeler_elms_concat.extend(calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval))
        
        consensus = calc_consensus(labeler_states.swapaxes(0,1)) #has -1 in locations which are not consensual, ie at least one person disagrees (case 3)
        print('calculating with consensus opinion')
        print(sum(consensus == -1))
        consensus_concat.extend(consensus)
        
        # majority = calc_mode_remove_consensus(labeler_states.swapaxes(0,1)) #has -2 in locations of consensus, -1 in locations of total disagreement (case 2)
        # print('calculating with majority opinion removing consensus')
        # print(sum(majority == -1), sum(majority == -2), sum(majority > 0))
        
        # mode_labeler_states = ground_truth
        mask1 = np.where(ground_truth != -1)[0]
        temp2 = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        # assert len(temp1) == len(temp2)
        mask2 = np.where(temp2 != -1)[0]
        mask = list(set(mask1) & set(mask2))
        ground_truth = ground_truth[mask]
        pred_states_disc = pred_states_disc[mask]
                
        
        # fshots[shot] = fshot
        # pdf_save_dir = model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + 'shot_simp' + shot + '.pdf'
        # plot_shot_simplified(shot, fshot.copy(), pdf_save_dir)
        
        if not os.path.isdir(pdf_save_dir):
            os.makedirs(pdf_save_dir)
        plot_fname = pdf_save_dir + '/' + 'shot' + shot + '.pdf'
        plot_shot_cnn_viterbi_full(shot, fshot.copy(), plot_fname, dice_cf, k_st)
        
        print('TCV_'  + str(shot) + '_CNN_det.csv')
        fshot_csv_fname = model_dir + '/' + epoch_to_predict + '/detector_csv_out/' + exp_arg + '/'
        if not os.path.isdir(fshot_csv_fname):
            os.makedirs(fshot_csv_fname)
            
        fshot.to_csv(columns=['time', 'IP', 'FIR', 'PD', 'DML', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'],
                        path_or_buf= fshot_csv_fname + '/TCV_'  + str(shot) + '_CNN_det.csv', index=False)
    
    ground_truth_concat = np.asarray(ground_truth_concat)
    consensus_concat = np.asarray(consensus_concat)
    states_pred_concat = np.asarray(states_pred_concat)
    labeler_elms_concat = np.asarray(labeler_elms_concat)
    k_indexes = np.asarray(k_indexes)
    
    ground_truth_mask = np.where(ground_truth_concat!=-1)[0]
    elm_label_mask = np.where(labeler_elms_concat!=-1)[0]
    mask = list(set(ground_truth_mask) & set(elm_label_mask))

    ground_truth_concat = ground_truth_concat[ground_truth_mask]
    states_pred_concat = states_pred_concat[ground_truth_mask]
    consensus_concat = consensus_concat[ground_truth_mask] #should stay the same, as consensus is subset of ground truth
    
    print('ground_truth_concat', ground_truth_concat.shape)
    print('states_pred_concat', states_pred_concat.shape)
    
    print(k_statistic(states_pred_concat, ground_truth_concat))
    print(k_statistic(consensus_concat, ground_truth_concat))

    title = ''
    concat_elms = np.asarray(concat_elms)
    concat_elm_labels = np.asarray(concat_elm_labels)
    roc_curve = get_roc_curve(concat_elms, concat_elm_labels, thresholds, gaussian_hinterval=10, signal_times=[])
    roc_fname = pdf_save_dir + epoch_to_predict + exp_arg + 'roc_curve.pdf'
    plot_roc_curve(roc_curve, thresholds, roc_fname, title)
    
    histo_fname = pdf_save_dir + epoch_to_predict + exp_arg + 'k_ind_histogram.pdf'
    
    plot_kappa_histogram(k_indexes, histo_fname, title)
    
    
    fpath = model_dir + '/' + epoch_to_predict + '/' + exp_arg + 'k_ind_sorted_scores_'
    out_sorted_scores(k_indexes_dic, fpath)
    
 
if __name__ == '__main__':
    main()