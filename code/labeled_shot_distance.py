import sys
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
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
from plot_shot_versions import *

def main():
    compare(sys.argv)
    
    
def compare(args):
    print('------------STARTING------------')
    # gaussian_time_window = 1e-3
    # signal_sampling_rate = 1e4
    # gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    # print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    # exp_arg = args[3]
    # shots = args[2].split(",")
    s_ids = (61057,)
             # 57103,26386,33459,43454,34010,32716,32191,61021,
             #    30197,31839,60097,60275,32195,32911,59825,53601,34309,30268,33638,
             #    31650,31554,42514,39872,26383,48580,62744,32794,30310,31211,31807,
             #    47962,57751,31718,58460,57218,33188,56662,33271,30290,
             #    33281,30225,58182,32592, 30044,30043,29511,33942,45105,52302,42197,30262,42062,45103,33446,33567)
    # s_ids = (60097,)
    # s_ids = (60097,47962,39872,42062,34010,58460,31839,32592,32794,61057) #61057
    shots = [str(i) for i in s_ids]
    print('no shots', len(shots))
    # print(shots, len(shots))
    # labeler = 'ffelici' #labit
    # data_dir = '../../data3/labeled/' + labeler
    data_dir = '../../data4/labeled/'
    labelers = args[1].split(",")
    print('labelers', labelers)
    # X_scalars_test = []
    # fshots = {}
    # conv_window_size = 40
    # num_classes = 3
    # no_input_channels = 3
    dice_cfs = {}
    
    sids = []
    # shot_signals_t =
    lstm_exp = args[2].split(',')
    for i, shot in zip(range(len(shots)), shots):
        fshots = {}
        fshots_times = {}
        fshot_time_sets = {}
        fshots_states = []
        fshots_elms = []
        for labeler in labelers:
            print('Reading shot', shot, 'from labeler', labeler)
            print(labeler+'-'+shot)
            # continue
            # try:
            # fshot = pd.read_csv(data_dir +'/test/TCV_'  + str(shot) + '_signals.csv', encoding='utf-8')
            
            if labeler == 'lstm':
                fshot, fshot_times = load_fshot_from_classifier(shot + '-' + lstm_exp[1], '../../data4/LSTM_predicted/'+lstm_exp[0] + '/')
            else:
                fshot, fshot_times = load_fshot_from_labeler(shot + '-' + labeler, data_dir)
            fshots[labeler] = fshot
            fshots_times[labeler] = set(fshot_times.round(5))
            # print(sorted(fshots_times[labeler])[:10])        
            # print(sorted(fshots_times[labeler])[-10:])
            
        t_itsc = sorted(set.intersection(*fshots_times.values()))
        # print(fshots_times.values())
        for labeler in labelers:
            fshot = fshots[labeler]
            fshot = fshot[fshot['time'].round(5).isin(t_itsc)]
            fshot = normalize_signals_mean(fshot)
            fshots[labeler] = fshot
            if labeler == 'lstm':
                fshots_states += [fshot['LHD_det'].values]
                fshots_elms += [fshot['ELM_det'].values]
            else:
                fshots_states += [fshot['LHD_label'].values]
                fshots_elms += [fshot['ELM_label'].values]
            sids += [shot]
        # val_itsc = sorted(set.intersection(*fshots.values().))
        # print(val_itsc)
        
        fshot_pd_val = fshots['ffelici'].PD.values
        for l in labelers:
            # print(l)
            # print(fshots[l].PD.values[:5], len(fshots[l].PD.values))
            # plt.plot(fshot_pd_val, fshots[l].PD.values)
            # plt.show()
            assert np.array_equal(np.round(fshot_pd_val, 7), np.round(fshots[l].PD.values, 7))
        
        plot_shot_versions(shot_t=fshot.time.values, shot_sig=fshot.PD.values, fshots_states=fshots_states, fshots_elms=fshots_elms, save_dir=data_dir + 'shot_plots_by_dist/', shot_id=shot, labelers = labelers)
    
    
    
    
if __name__ == '__main__':
    main()