import sys
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from plot_shot_results import plot_shot_cnn, plot_shot_cnn_full
from collections import OrderedDict
import csv
from cnn_train import *

def main():
    predict(sys.argv)
    
    
def predict(args):
    print('------------STARTING------------')
    gaussian_time_window = 1e-3
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    exp_arg = args[3]
    # print(args)
    shots = args[6].split(",")
    # print(shots, len(shots))
    # labeler = 'ffelici' #labit
    labelers = args[4].split(',')
    print(labelers)
    data_dir = './data/labeled/'
    # data_dir = '../../data4/labeled/' + labeler
    X_scalars_test = []
    fshots = {}
    conv_window_size = 200
    num_classes = 3
    no_input_channels = 3
    conv_w_offset = int(args[5])
    for i, shot in zip(range(len(shots)), shots):
        print('Reading shot', shot)
        # continue
        # try:
        # fshot = pd.read_csv(data_dir +'/test/TCV_'  + str(shot) + '_signals.csv', encoding='utf-8')
        fshot = pd.read_csv(data_dir + labelers[0] + '/TCV_'  + str(shot) + '_' + labelers[0] + '_labeled.csv', encoding='utf-8')
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
    # X_scalars_test = np.asarray(X_scalars_test)
    # print('X_scalars_test', X_scalars_test.shape)
    model_dir = args[1]
    epoch_to_predict = args[2]
    # model_path = model_dir + '/model_weights.h5'
    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5' 
    model = cnn(conv_w_size=conv_window_size)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # model_path = model_dir + '/model_checkpoints/modelold.h5' 
    model.load_weights(model_path)
    
    print('Loading from,', model_path)
    print('Predicting on shot(s)...')
    pred_transitions = []
    pred_elms = []
    dice_cfs = []
    conf_mats = {}
    for t in get_trans_ids():
        conf_mats[t] = [0,0,0,0]
    conf_mets = []
    dice_cfs_dic = {}
    for k in range(len(shots)):
        print('Predicting shot ' + str(shots[k]), X_scalars_test[k].shape)
        # plt.plot(X_scalars_test[k][0,0,2])
        # plt.plot(X_scalars_test[k][:,10,2])
        # plt.show()
        elms, transitions = model.predict(
                                {'conv_input':np.asarray(X_scalars_test[k])},
                                batch_size=128,
                                verbose=1
                                )
        print('Predicted sequence length is', str(transitions.shape), str(elms.shape))
        # print(elms[0])
        print(elms.shape, transitions.shape)
        # plt.plot(elms)
        # plt.plot(transitions)
        # plt.show()
        pred_transitions += [transitions]
        pred_elms += [elms]
    
    
    # pred_transitions = np.asarray(pred_transitions)
    # pred_elms = np.asarray(pred_elms)
    exit(0)
    #remaining code is up for deletion
    
    # CONVERT TO DISCRETE LABELS and save
    collapsed_shots = []
    collapsed_shots_labels = []
    print('Post processing, saving .csv file(s)...')
    
    for i, shot in enumerate(shots):
        labeler_states = []
        labeler_transitions = {}
        labeler_elms = []
        print('----------------------------------------SHOT', str(shot), '-----------------------------------------')
        # fshot = fshots[shot].reset_index()
        # fshot = fshots[shot].reset_index()
        labeler = labelers[0]
        fshot = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
        fshot = remove_current_30kA(fshot)
        fshot = remove_no_state(fshot)
        fshot = remove_disruption_points(fshot)
        fshot = fshot.reset_index()
        fshot = fshot[conv_window_size-conv_w_offset:len(fshot)-conv_w_offset]#careful here
        fshot = fshot.reset_index(drop=True)
        # print(fshot)
        # plt.plot(shot_df.FIR.values)
        # plt.plot(shot_df.PD.values)
        # plt.plot(shot_df.DML.values)
        # plt.plot(shot_df.LHD_label.values)
        # plt.plot(shot_df.ELM_label.values)
        # plt.show()
        
        fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
        intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(fshot.time.values,5))), 5)
        fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]
        fshot_sliced = fshot_sliced.reset_index(drop=True)
        # fshot_sliced = state_to_trans_event_disc(fshot_sliced, gaussian_hinterval)
        
        # labeler_states += [fshot_sliced['LHD_label'].values]
        labeler_transitions[labeler] = state_to_trans_event_disc(fshot_sliced, gaussian_hinterval)
        labeler_elms += [fshot_sliced['ELM_label'].values]
        labeler_states += [fshot_sliced['LHD_label'].values]
        if len(labelers) > 1:
            for k, labeler in enumerate(labelers[1:]):
                fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
                intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(fshot.time.values,5))), 5)
                fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]
                labeler_transitions[labeler] = state_to_trans_event_disc(fshot_sliced, gaussian_hinterval)
                labeler_elms += [fshot_sliced['ELM_label'].values]
                labeler_states += [fshot_sliced['LHD_label'].values]
        labeler_states = np.asarray(labeler_states)
        labeler_elms = np.asarray(labeler_elms)
        
        pred_trans = pred_transitions[i][:len(fshot_sliced)]
        trans_ids = get_trans_ids() + ['no_trans']
        # print(fshot_sliced)
        for t, t_id in enumerate(trans_ids):
            collapsed_trans = []
            for lab in labeler_transitions.values():
                # print(lab.shape)
                collapsed_trans += [lab[t_id]]
            collapsed_trans = np.asarray(collapsed_trans)
            # print(collapsed_trans.shape)
            pred_trans_disc = event_cont_to_disc(pred_trans[:, t], threshold = .5)
            fshot[t_id + '_lab'] = calc_elm_mode(collapsed_trans.swapaxes(0,1), gaussian_hinterval)
            fshot[t_id + '_det'] = pred_trans_disc
            fshot[t_id + '_det_prob'] = pred_trans[:, t]
        
        fshot['ELM_label'] = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
            
        lhd_det = det_trans_to_state(fshot)
        fshot['LHD_det'] = pd.Series(lhd_det)
        
        mode_labeler_states = calc_mode(labeler_states.swapaxes(0,1))
        mask = np.where(mode_labeler_states != -1)[0]
        dice_cf = dice_coefficient(lhd_det[mask], mode_labeler_states[mask])
        k_st = k_statistic(lhd_det[mask], mode_labeler_states[mask])
        
        elm_prob = pred_elms[i][:len(fshot_sliced),0]
        # temp1 = pred_elms[i][0,:,0][:len(fshot_sliced)]
        # temp2 = fshot['ELM_label'].values
        # mask = np.where(temp2 != -1)[0]
        # assert len(temp1) == len(temp2)
        # collapsed_shots.extend(temp1[mask])
        # collapsed_shots_labels.extend(temp2[mask])
        
        # plt.plot(range(len(fshot_sliced)),fshot_sliced['PD'].values)
        # plt.plot(range(len(fshot_sliced)),fshot_sliced[get_trans_ids()].values)
        # plt.legend(['PD'] + get_trans_ids())
        # plt.show()
        # pred_trans_disc = np.argmax(pred_transitions[i][0,:], axis=1)
        # pred_states_disc = pred_states_disc[:len(fshot_sliced)]
        # pred_states_disc += 1 #necessary because argmax returns 0 to 2, while we want 1 to 3!
        # dice_cf = dice_coefficient(pred_states_disc, fshot_sliced['LHD_label'].values)
        temp1 = pred_trans[:, :-1] #remove no_trans
        temp2 = fshot_sliced[get_trans_ids()].values
        assert len(temp1) == len(temp2)
        collapsed_shots.extend(temp1)
        collapsed_shots_labels.extend(temp2)
        # plt.plot(fshot['time'].values,fshot_sliced['HL'].values)
        # plt.legend(['PD', 'HL'])
        # plt.show()
        
        pdf_save_dir = model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + 'shot' + shot + '.pdf'
        # pred_elms_disc = elms_cont_to_disc(pred_elms[i][0,:,0][:len(fshot_sliced)], threshold = best_threshold)
        # print(pred_elms[i].shape)
        fshot['ELM_prob'] = elm_prob
        plot_shot_cnn_full(shot, fshot.copy(), pdf_save_dir, dice_cf, k_st)
        # plot_shot(shot, fshot.copy(), pdf_save_dir)
    
    collapsed_shots = np.asarray(collapsed_shots)
    collapsed_shots_labels = np.asarray(collapsed_shots_labels)
    # exit(0)
    for t_id, trans in enumerate(get_trans_ids()): # +get_trans_ids()
        print(trans, 'transition')
        
        thresholds = [0., .25, .5, .75, 1.]
        # thresholds =[.5,]
        roc_curve = get_roc_curve(collapsed_shots[:, t_id], collapsed_shots_labels[:, t_id], thresholds, gaussian_hinterval=10, signal_times=[])
        # print('roc curve .5', roc_curve[.5])
        threshold, dist = get_roc_best(roc_curve)
        fs = matplotlib.rcParams['font.size']
        matplotlib.rcParams.update({'font.size': 18})
        f = PdfPages(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + epoch_to_predict + exp_arg +  trans + '_roc_curve.pdf')
        fig = plt.figure(figsize = (7, 7))
        fig.suptitle(trans)
        ax = fig.add_subplot(111)
        fprs = []
        tprs = []
        for t in thresholds:
            fpr, tpr = roc_curve[t]
            fprs += [fpr]
            tprs += [tpr]
            if t in (0, 1, threshold):
                ax.annotate(t, (fpr, tpr))
            # p.scatter(fpr, tpr, alpha=1.0, c='blue', edgecolors='none')
        ax.plot(fprs, tprs, 'o-')
        # ax.set_xlim([-.005,.05])
        ax.set_xlim([-.11,1.1])
        ax.set_ylim([-.11,1.1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        # labels = {str(threshold): dis, '0':roc_curve[0], '1':roc_curve[1]}
        # labels=[dis, ]
        # for t in labels.keys():
        #     ax.annotate(d, (fprs[t], tprs[t]))
        print(threshold, roc_curve[threshold])
        # matplotlib.rcParams.update({'font.size': fs})
        # plt.show()
        fig.savefig(f, format='pdf')
        f.close()
        
            
            # best_threshold = .5
            # pred_trans_disc = event_cont_to_disc(temp1, threshold = best_threshold)
            # true_positives, false_positives, true_negatives, false_negatives = conf_matrix(pred_trans_disc, temp2, gaussian_hinterval=10, signal_times=[])
            # conf_mat = np.asarray([true_positives, false_positives, true_negatives, false_negatives])
            # conf_mats[trans] += [conf_mat]
            
            # dice_cfs += [dice_cf]
            # dice_cfs_dic[shot] = dice_cf
            # conf_mets += [conf_met]
            # print(fshot)
            # fshot = fshot.reset_index()
            # fshot['L_prob'] = pred_states[i][0,:, 0]
            # fshot['H_prob'] = pred_states[i][0,:, 2]
            # fshot['D_prob'] = pred_states[i][0,:, 1]
            
            
            
            # fshot['ELM_prob'] = temp1
            # fshot['ELM_det'] = pred_elms_disc
            # fshot['LHD_det'] = pred_states_disc
            
            # print(fshot_sliced.LHD_label)
            
            # print(len(pred_states_disc[:len(fshot_sliced)]), len(fshot_sliced))
            # print('Dice coeficients for this shot (L-D-H-total):', dice_cf)
            # print('Confusion matrix for this shot, calculated with threshold of ' + str(best_threshold) + ' (TP, FP, TN, FN):', conf_mat.astype(int))
            # plt.plot(fshot['time'].values,fshot['PD'].values, label='PD')
            # # plt.plot(fshot['time'].values,fshot['ELM_det'].values)
            # plt.plot(fshot['time'].values,temp1, label=trans)
            # # plt.legend(['PD', 'HL'])
            # plt.legend()
            # plt.show()
            # 
        
        # temp1 = pred_elms[i][:len(fshot_sliced),0]
        # temp2 = fshot_sliced['ELM_label'].values
        # 
        # best_threshold = .5
        # pred_elms_disc = elms_cont_to_disc(temp1, threshold = best_threshold)
        # true_positives, false_positives, true_negatives, false_negatives = elm_conf_matrix(pred_elms_disc, temp2, gaussian_hinterval=10, signal_times=[])
        # conf_mat = np.asarray([true_positives, false_positives, true_negatives, false_negatives])
        # conf_mats += [conf_mat]
        # print('ELM: Confusion matrix for this shot, calculated with threshold of ' + str(best_threshold) + ' (TP, FP, TN, FN):', conf_mat.astype(int))
        # 
        # fshot['ELM_prob'] = temp1
        # fshot['ELM_det'] = pred_elms_disc
        
        # plt.plot(fshot['time'].values,fshot['PD'].values)
        # # plt.plot(fshot['time'].values,fshot['ELM_det'].values)
        # plt.plot(fshot['time'].values,pred_transitions[i][:len(fshot_sliced)])
        # plt.legend(['PD', 'ELM',] + get_trans_ids())
        # plt.show()
        # pdf_save_dir = model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + 'shot' + shot + '.pdf'
        # plot_shot(shot, fshot.copy(), pdf_save_dir)
        
        
        # print('TCV_'  + str(shot) + '_LSTM_det.csv')
        # fshot.to_csv(columns=['time', 'IP', 'FIR', 'PD', 'DML', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'],
        #                   path_or_buf=data_dir + '/LSTM_predicted/' + model_dir + '/' + epoch_to_predict +  '/TCV_'  + str(shot) + '_LSTM_det.csv', index=False)
    # exit(0)
    collapsed_shots = np.asarray(collapsed_shots)
    # collapsed_shots = collapsed_shots.reshape((len(shots) * collapsed_shots.shape[1],) + collapsed_shots.shape[2:])
    # # print(collapsed_shots.shape)
    collapsed_shots_labels = np.asarray(collapsed_shots_labels)
    # collapsed_shots_labels = collapsed_shots_labels.reshape((len(shots) * collapsed_shots_labels.shape[1],) + collapsed_shots_labels.shape[2:])
    print(collapsed_shots.shape, collapsed_shots_labels.shape)
    # for i, shot in enumerate(shots):
    thresholds = [.25, .5, .75,]
    thresholds = np.arange(5, 100, step=5)/100
    thresholds = np.arange(105, step=5)/100
    roc_curve = get_roc_curve(collapsed_shots, collapsed_shots_labels, thresholds, gaussian_hinterval=10, signal_times=[])
    threshold, dist = get_roc_best(roc_curve)
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 18})
    f = PdfPages(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + epoch_to_predict + exp_arg + 'roc_curve.pdf')
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111)
    fprs = []
    tprs = []
    for t in thresholds:
        fpr, tpr = roc_curve[t]
        fprs += [fpr]
        tprs += [tpr]
        if t in (0, 1, threshold):
            ax.annotate(t, (fpr, tpr))
        # p.scatter(fpr, tpr, alpha=1.0, c='blue', edgecolors='none')
    ax.plot(fprs, tprs, 'o-')
    # ax.set_xlim([-.005,.05])
    ax.set_xlim([-.11,1.1])
    ax.set_ylim([-.11,1.1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # labels = {str(threshold): dis, '0':roc_curve[0], '1':roc_curve[1]}
    # labels=[dis, ]
    # for t in labels.keys():
    #     ax.annotate(d, (fprs[t], tprs[t]))
    print(roc_curve)
    matplotlib.rcParams.update({'font.size': fs})
    # plt.plot()
    
    fig.savefig(f, format='pdf')
    f.close()
    # plt.show()
    # exit(0)
        
    st_and_mean = ['Low', 'Dither', 'High', 'Mean']
    for k, state in enumerate(st_and_mean):
        print('Shots ordered by lowest do highest dice, sorted by ' + state)
        dd = OrderedDict(sorted(dice_cfs_dic.items(), key=lambda x: x[1][k]))
        with open(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '_sorted_scores_' + state + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['shot', 'Low DC', 'Dither DC', 'High DC', 'Mean DC'])
            w.writeheader()
            for key, val in dd.items():
                # print(key, val[k])
                row = {'shot': key}
                for c_l, l in enumerate(st_and_mean):
                    # print(val, l, c_l)
                    # print(l + ' DC', val[c_l])
                    row[l + ' DC'] = val[c_l]
                    
                w.writerow(row)
    

    # for i, shot in enumerate(shots):
    #     thresholds = [.25, .5, .75,]
    #     # pred_elms = np.asarray(pred_elms).reshape(len(shots) * )
    #     temp = pred_elms[i][0,:,0][:len(fshot_sliced)]
    #     collapsed_shots += [temp]
    #     
    #     roc_curve = get_roc_curve(pred_elms[i][0,:,0][:len(fshot_sliced)], fshot_sliced['ELM_label'].values, thresholds, gaussian_hinterval=10, signal_times=[])
    # 
    #             
    dice_cfs = np.asarray(dice_cfs)
    # print(dice_cfs.shape)
    f = PdfPages(model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + epoch_to_predict + 'histogram.pdf')
    # matplotlib.rcParams["font.size"] = 18
    fig = plt.figure(figsize = (10, 5))
    
    for k, state in enumerate(st_and_mean):
        dice_state = dice_cfs[:, k]
        print(dice_state)
        p = fig.add_subplot(1,len(st_and_mean),k+1)
        p.hist(dice_state, bins=10, range=(0, 1))
        # p.hist(dice_state, bins=, range=(0, 1))
        # fig.suptitle(state)
        p.set_title(state)
        p.set_ylim(bottom = None, top=dice_state.shape[0] + 1)
        # p.set_xticks([0, 0.5, 1.])
        # p.set_xlabel('Dice Coefficient Scores ')
        # p.set_ylabel('Number of shots')
        # p.xlabel('Dice Coefficient Scores ')
        # p.ylabel('Number of shots')
        # plt.tight_layout()
        fig.suptitle('Dice coeficient histograms per state')
        fig.text(.06, .5, 'Frequency', ha='center', va='center', rotation='vertical')
        fig.text(.5, 0.025, 'Dice Coefficient Scores', ha='center', va='center', rotation='horizontal')
        plt.tight_layout()
        hist, bin_edges = np.histogram(dice_state, bins=[0., 0.5, 0.75, 0.9, 0.95, 1])
        print('histogram for', state, hist)
    
    fig.savefig(f, format='pdf')
    f.close()
    print('mean dice cfs', np.mean(dice_cfs, axis=0))
    temp = np.sum(np.asarray(conf_mats), axis=0)
    print('sum conf mats', temp)
    # print('mean conf mets', np.mean(np.asarray(conf_mets), axis=0))
    print('mean conf mets', conf_metrics(temp[0], temp[1], temp[2], temp[3]))
    
if __name__ == '__main__':
    main()