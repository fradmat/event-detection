import numpy as np
import pandas as pd
import os
# from plot_shots_and_events import get_window_plots_nn_data, get_window_plots, save_signal_plot_nn_data
# from sliding_windows import *
import datetime
# from shot_correction import get_event_type
# from plot_shots_and_events import save_shot_plot
# from matplotlib.backends.backend_pdf import PdfPages
# import pickle
import matplotlib.transforms as mtransforms
import sys
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from helper_funcs import *
import matplotlib
from matplotlib.gridspec import GridSpec
from window_functions import get_raw_signals_in_window
import seaborn as sns
from matplotlib.colors import ListedColormap

def main():
    plot_shot()
    
    
def plot_shot_cnn(shot_id, fshot, directory):
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    # states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    # print('states', states.shape)
    
        
    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    for k in range(len(fshot)//wd + 1):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print 'start', start
        finish = (k+1) * wd
        if k == len(fshot)//wd:
            # print('last')
            finish = len(fshot) - 1
        #     # print fshot[start:finish].time.values
        #     print len(fshot[start:finish].time.values)
        # print len(fshot[start:finish].time.values)
        # print(start, finish)
        # this_wd = fshot[start:finish]
        fig = plt.figure(figsize = (19, 20))
        plt.suptitle('shot #' + str(f))
        
        lab = fshot.LHD_label.values
        fpad_max = np.max(np.concatenate((fshot.FIR.values, fshot.PD.values)))
        fir_dml = fig.add_subplot(16,1,(1,2))
        fir_dml.grid()
        fir_dml.set_title('FIR and DML signals (normalized) + labels (state and ELMs), whole shot')
        fir_dml.plot(fshot.time.values, fshot.FIR.values, fshot.time.values, fshot.DML.values)
        fir_dml.set_xlabel('time, s')
        fir_dml.legend(['FIR', 'DML'], loc=2, prop={'size': leg_size})
        fir_dml.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        fpad_max = np.max(np.concatenate((fshot.FIR.values, fshot.DML.values)))
        fpad_min = np.min(np.concatenate((fshot.FIR.values, fshot.DML.values)))
        fir_dml.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2)
        
        pd_ip = fig.add_subplot(16,1,(3,4))
        pd_ip.grid()
        pd_ip.set_title('PD and IP signals (normalized) + labels (state and ELMs), whole shot')
        pd_ip.plot(fshot.time.values, fshot.PD.values, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip.set_xlabel('time, s')
        pd_ip.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        pd_ip.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2)
        
        
        # 
        fir = fig.add_subplot(16,1,(5,6))
        fir.set_title('FIR (electron density) values, normalized')
        # trans = mtransforms.blended_transform_factory(fir.transData, fir.transAxes)
        
        fir.plot(fshot[start:finish].time.values, fshot[start:finish].FIR.values)
        min_d, max_d = np.min(fshot[start:finish].FIR.values), np.max(fshot[start:finish].FIR.values)
        fir.set_xlabel('time, s')
        fir.set_ylim(.95*min_d, 1.05*max_d)
        fir.grid()
        fir.legend(['FIR'], loc=2, prop={'size': leg_size}, ncol=2)
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(16,1,(7,8))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values)
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        phd.legend(['PD'], loc=2, prop={'size': leg_size}, ncol=2)
        
        dml = fig.add_subplot(16,1,(9,10))
        dml.set_title('DML (magnetic field) values, normalized')
        dml.plot(fshot[start:finish].time.values, fshot[start:finish].DML.values)
        dml.set_xlabel('time, s')
        min_d, max_d = np.min(fshot[start:finish].DML.values), np.max(fshot[start:finish].DML.values)
        dml.set_ylim(.95*min_d, 1.05*max_d)
        dml.grid()
        dml.legend(['DML'], loc=2, prop={'size': leg_size}, ncol=2)
        
        # ip = fig.add_subplot(16,1,(11,12))
        # ip.set_title('ip values, normalized')
        # ip.plot(fshot[start:finish].time.values, fshot[start:finish].IP.values)
        # ip.set_xlabel('time, s')
        # min_d, max_d = np.min(fshot[start:finish].IP.values), np.max(fshot[start:finish].IP.values)
        # ip.set_ylim(.95*min_d, 1.05*max_d)
        # ip.grid()
        # ip.legend(['IP'], loc=2, prop={'size': leg_size}, ncol=2)
        
        
        elm = fig.add_subplot(16,1,(11,12))
        elm.plot(fshot[start:finish].time.values, elms[start:finish], color = 'b')
        fshot_cl = fshot[start:finish]
        elm.vlines(x = fshot_cl[fshot_cl.ELM_label > .5].time.values, color = 'b', ymin = 0, ymax=1, alpha=0.2)
        elm.set_xlabel('time, s')
        elm.set_title('elm labels (vertical lines) and LSTM elm output (continuous line)')
        elm.legend(['ELM'], loc=2, prop={'size': leg_size}, ncol=2)
        elm.set_ylim([-0.1,1.1])
        elm.grid()
        
        t_list = get_trans_ids() + ['no_trans']
        c = ['r','g','y',colors['teal'],'c','m',colors['black']]
        t_list = [t + '_det' for t in t_list]
        transitions_det = fshot[t_list]
        trans = fig.add_subplot(16,1,(13,14))
        trans.set_xlabel('time, s')
        trans.set_title('state labels (solid color background) and LSTM state outputs (colored lines)')
        for t, t_ind in enumerate(t_list):
            # print(c[t])
            trans.plot(fshot[start:finish].time.values, transitions_det[start:finish][t_ind].values, color = c[t])
        trans.legend(get_trans_ids() + ['no_trans'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        
        t_list = get_trans_ids()
        t_list = [t + '_lab' for t in t_list]
        fshot_sl = fshot[start:finish]
        for t, t_ind in enumerate(t_list):
            # print(fshot_sl[t_ind].shape)
            # print(fshot_sl[fshot_sl[t_ind].values > .5].time.shape)
            trans.vlines(x = fshot_sl[fshot_sl[t_ind].values > .01].time.values, color = c[t], ymin = 0, ymax=1, alpha=1, linewidth=3)
        
        trans.set_ylim([-0.1,1.1])
        trans.grid()
        
        state = fig.add_subplot(16,1,(13,14))
        state.plot(fshot[start:finish].time.values, states[start:finish,0], color = colors['green']) #L Mode
        
        state.plot(fshot[start:finish].time.values, states[start:finish,1], color = colors['gold']) #d Mode
        state.plot(fshot[start:finish].time.values, states[start:finish,2], color = colors['red']) #H Mode
        state.set_xlabel('time, s')
        state.set_title('state labels (solid color background) and LSTM state outputs (colored lines)')
        state.legend(['Low','Dither', 'High'], loc=2, prop={'size': leg_size}, ncol=2) #,'H', 'D'
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2)
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2)
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2)
        
        plt.tight_layout()
        fig.savefig(test, format='pdf')
        plt.close('all')
    test.close()
    sys.stdout.flush()
    
def plot_shot(shot_id, fshot, directory):
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print 'start', start
        finish = (k+1) * wd
        # if finish == len(fshot)
        # print(k, start, finish)
        # if k == len(fshot)//wd:
        if k == no_k - 1:
            # print('last')
            finish = len(fshot) - 1
        fig = plt.figure(figsize = (19, 20))
        plt.suptitle('#' + str(f))
        
        lab = fshot.LHD_label.values
        fpad_max = np.max(np.concatenate((fshot.FIR.values, fshot.PD.values)))
        fir_dml = fig.add_subplot(16,1,(1,2))
        fir_dml.grid()
        fir_dml.set_title('FIR and DML signals (normalized) + labels (state and ELMs), whole shot')
        fir_dml.plot(fshot.time.values, fshot.FIR.values, fshot.time.values, fshot.DML.values)
        fir_dml.set_xlabel('time, s')
        fir_dml.legend(['FIR', 'DML'], loc=2, prop={'size': leg_size})
        fir_dml.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        fpad_max = np.max(np.concatenate((fshot.FIR.values, fshot.DML.values)))
        fpad_min = np.min(np.concatenate((fshot.FIR.values, fshot.DML.values)))
        fir_dml.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2)
        fir_dml.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2)
        
        pd_ip = fig.add_subplot(16,1,(3,4))
        pd_ip.grid()
        pd_ip.set_title('PD and IP signals (normalized) + labels (state and ELMs), whole shot')
        pd_ip.plot(fshot.time.values, fshot.PD.values, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip.set_xlabel('time, s')
        pd_ip.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        pd_ip.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2)
        pd_ip.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2)
        
        
        # 
        fir = fig.add_subplot(16,1,(5,6))
        fir.set_title('FIR (electron density) values, normalized')
        # trans = mtransforms.blended_transform_factory(fir.transData, fir.transAxes)
        
        fir.plot(fshot[start:finish].time.values, fshot[start:finish].FIR.values)
        min_d, max_d = np.min(fshot[start:finish].FIR.values), np.max(fshot[start:finish].FIR.values)
        fir.set_xlabel('time, s')
        fir.set_ylim(.95*min_d, 1.05*max_d)
        fir.grid()
        fir.legend(['FIR'], loc=2, prop={'size': leg_size}, ncol=2)
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(16,1,(7,8))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values)
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        phd.legend(['PD'], loc=2, prop={'size': leg_size}, ncol=2)
        
        dml = fig.add_subplot(16,1,(9,10))
        dml.set_title('DML (magnetic field) values, normalized')
        dml.plot(fshot[start:finish].time.values, fshot[start:finish].DML.values)
        dml.set_xlabel('time, s')
        min_d, max_d = np.min(fshot[start:finish].DML.values), np.max(fshot[start:finish].DML.values)
        dml.set_ylim(.95*min_d, 1.05*max_d)
        dml.grid()
        dml.legend(['DML'], loc=2, prop={'size': leg_size}, ncol=2)
        
        
        elm = fig.add_subplot(16,1,(11,12))
        elm.plot(fshot[start:finish].time.values, elms[start:finish], color = 'b')
        fshot_cl = fshot[start:finish]
        elm.vlines(x = fshot_cl[fshot_cl.ELM_label > .5].time.values, color = 'b', ymin = 0, ymax=1, alpha=0.2)
        elm.set_xlabel('time, s')
        elm.set_title('elm labels (vertical lines) and LSTM elm output (continuous line)')
        elm.legend(['ELM'], loc=2, prop={'size': leg_size}, ncol=2)
        elm.set_ylim([-0.1,1.1])
        elm.grid()
        
        
        state = fig.add_subplot(16,1,(13,14))
        state.plot(fshot[start:finish].time.values, states[start:finish,0], color = colors['green']) #L Mode
        
        state.plot(fshot[start:finish].time.values, states[start:finish,1], color = colors['gold']) #d Mode
        state.plot(fshot[start:finish].time.values, states[start:finish,2], color = colors['red']) #H Mode
        state.set_xlabel('time, s')
        state.set_title('state labels (solid color background) and LSTM state outputs (colored lines)')
        state.legend(['Low','Dither', 'High'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2)
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2)
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2)
        # state.legend(['low', 'dither', 'high'], loc=2)
        state.set_ylim([-0.1,1.1])
        state.grid()
        plt.tight_layout()
        fig.savefig(test, format='pdf')
        plt.close('all')
    test.close()
    sys.stdout.flush()
    
def plot_shot_cnn_full(shot_id, fshot, directory, dice_cf, k_statistic):
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    # states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print 'start', start
        finish = (k+1) * wd
        # if finish == len(fshot)
        # print(k, start, finish)
        # if k == len(fshot)//wd:
        if k == no_k - 1:
            # print('last')
            finish = len(fshot) - 1
        # if k + 1 == len(fshot)//wd:
        #     break
        
            
        #     # print fshot[start:finish].time.values
        #     print len(fshot[start:finish].time.values)
        # print len(fshot[start:finish].time.values)
        # print(k, start, finish)
        # this_wd = fshot[start:finish]
        fig = plt.figure(figsize = (19, 17))
        plt.suptitle('Shot #' + str(f) + '\n' +
                     'Dice coefficient values: L:' + str(dice_cf[0])  +
                     ' D:' + str(dice_cf[1]) +
                     ' H:' + str(dice_cf[2]) +
                     ' Mean value:' + str(dice_cf[3]) + '\n' +
                     'K-statistic values: L:' + str(k_statistic[0]) +
                     ' D:' + str(k_statistic[1]) +
                     ' H:' + str(k_statistic[2]) +
                     ' Mean value:' + str(k_statistic[3]),
                     fontsize = 'xx-large')
        
        lab = fshot.LHD_label.values
        pd_ip_lab = fig.add_subplot(10,1,(1,2))
        pd_ip_lab.grid()
        pd_ip_lab.set_title('PD signal (normalized) + labels (majority opinion), (state and ELMs), whole shot')
        pd_ip_lab.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_lab.set_xlabel('time, s')
        
        pd_ip_lab.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_lab.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2, label='L, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2, label='D, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2, label='H, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        pd_ip_lab.legend(loc=2, prop={'size': leg_size}, ncol=2)
        
        pd_ip_classifier = fig.add_subplot(10,1,(3,4))
        pd_ip_classifier.grid()
        pd_ip_classifier.set_title('PD signal (normalized) + Convnet classification (obtained through threshold + state machine) and ELM detection (threshold=.5), whole shot')
        pd_ip_classifier.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_classifier.set_xlabel('time, s')
        # pd_ip_classifier.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        pd_ip_classifier.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_classifier.vlines(x = fshot[fshot.ELM_prob > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 1, facecolor=colors['green'], alpha=0.2, label='Low, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 2, facecolor=colors['gold'], alpha=0.2, label='D, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 3, facecolor=colors['red'], alpha=0.2, label='H, CNN')
        pd_ip_classifier.legend(loc=2, prop={'size': leg_size})
        
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(10,1,(5,6))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values)
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        phd.legend(['PD'], loc=2, prop={'size': leg_size}, ncol=2)
        
        elm = fig.add_subplot(10,1,(7,8))
        elm.plot(fshot[start:finish].time.values, elms[start:finish], color = 'b', label='ELM, CNN')
        fshot_cl = fshot[start:finish]
        fshot_cl_wh = np.where(fshot_cl.ELM_label.values == 1)[0]
        # print(fshot_cl_wh, len(fshot_cl_wh))
        temp = np.zeros(len(fshot_cl)).astype(bool)
        if len(fshot_cl_wh) > 0:
            for ind in fshot_cl_wh:
                # print(ind)
                ids = np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)
                ids = np.clip(ids, a_min=0, a_max=min(len(fshot), len(temp)-1))
                # ids = np.clip(ids, a_min=0, a_max=len(temp))
                # temp.extend(list(np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)))
                temp[ids] = True
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= temp, facecolor = 'b', alpha=0.2, label='ELM, label')
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= fshot_cl.ELM_label.values == -1, facecolor = colors['lightgray'], alpha=0.8, label='No agreement, label')
        elm.set_xlabel('time, s')
        elm.set_title('elm labels (vertical lines) and direct CNN elm output (continuous line)')
        # elm.legend(['ELM'], loc=2, prop={'size': leg_size}, ncol=2)
        elm.legend(loc=2, prop={'size': leg_size})
        elm.set_ylim([-0.1,1.1])
        elm.grid()
        
        
        t_list = get_trans_ids()#+ ['no_trans']
        c = ['r','g','y',colors['teal'],'c','m',colors['black']]
        t_list = [t + '_det' for t in t_list]
        # transitions_det = fshot[t_list]
        trans = fig.add_subplot(10,1,(9,10))
        trans.set_xlabel('time, s')
        trans.set_title('state labels (solid color background), direct CNN transition outputs (colored lines), and thresholded CNN outputs(vertical bars)')
        fshot_sl = fshot[start:finish]
        # print(fshot_sl.columns)
        for t_ind, t in enumerate(t_list):
            # print(c[t])
            # trans.plot(fshot[start:finish].time.values, transitions_det[start:finish][t_ind].values, color = c[t])
            trans.vlines(x = fshot_sl[fshot_sl[t].values > .01].time.values, color = c[t_ind], ymin = 0, ymax=1, alpha=1, linewidth=3, label=t[:2] + ', CNN')
            trans.plot(fshot_sl.time.values, fshot_sl[t + '_prob'].values, color = c[t_ind])
        # trans.legend(get_trans_ids() + ['no_trans'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2, label='Low, label')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, label')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2, label='High, label')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        t_list = get_trans_ids()
        t_list = [t + '_lab' for t in t_list]
        # fshot_sl = fshot[start:finish]
        # for t, t_ind in enumerate(t_list):
        #     # print(fshot_sl[t_ind].shape)
        #     # print(fshot_sl[fshot_sl[t_ind].values > .5].time.shape)
        #     trans.vlines(x = fshot_sl[fshot_sl[t_ind].values > .01].time.values, color = c[t], ymin = 0, ymax=1, alpha=1, linewidth=3)
        trans.legend(loc=2, prop={'size': leg_size})
        trans.set_ylim([-0.1,1.1])
        trans.grid()
        
        
        # state = fig.add_subplot(12,1,(11,12))
        # # state.plot(fshot[start:finish].time.values, states[start:finish,0], color = colors['green'], label='Low, LSTM') #L Mode
        # # 
        # # state.plot(fshot[start:finish].time.values, states[start:finish,1], color = colors['gold'], label='Dither, LSTM') #d Mode
        # # state.plot(fshot[start:finish].time.values, states[start:finish,2], color = colors['red'], label='High, LSTM') #H Mode
        # state.set_xlabel('time, s')
        # state.set_title('state labels (solid color background, majority opinion) and direct LSTM outputs (colored lines)')
        # # state.legend(['Low','Dither', 'High'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        # state.fill_between(fshot[start:finish].time.values, 0., 1.,
        #                  where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2, label='Low, label')
        # state.fill_between(fshot[start:finish].time.values, 0., 1.,
        #                  where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, label')
        # state.fill_between(fshot[start:finish].time.values, 0., 1.,
        #                  where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2, label='High, label')
        # state.fill_between(fshot[start:finish].time.values, 0., 1.,
        #                  where= fshot[start:finish].LHD_label.values == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        # 
        # # state.legend(['low', 'dither', 'high'], loc=2)
        # state.legend(loc=2, prop={'size': leg_size}) #,'H', 'D'
        # state.set_ylim([-0.1,1.1])
        # state.grid()
        # 
        
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(test, format='pdf')
        plt.close('all')
        
        
    test.close()
    sys.stdout.flush()

# def plot_shot_full_seq2seq(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot, convolutional_stride,
#                           conv_w_size, block_size, fname, max_source_sentence_chars,
#                           max_train_target_words, look_ahead, num_channels, lhd_labels):
def plot_shot_full_seq2seq(shot_id, fshot, states_blocks, block_size, look_ahead, k_statistic, directory): #k_statistic
  
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # elms = fshot.ELM_prob.values
    # states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    # blocks_per_page = wd // block_size
    cs = [colors['gray'], colors['green'],colors['gold'],colors['red']]
    labels = ['no prediction', 'L, seq2seq', 'D, seq2seq', 'H, seq2seq']
    look_ahead_blocks = look_ahead // block_size
    initial_blocks = np.zeros(look_ahead_blocks)
    states_blocks = np.concatenate((initial_blocks, states_blocks))
    # print(initial_blocks)
    # print(states_blocks[:20])
    # exit(0)
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print(k)
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print('start', start)
        finish = (k+1) * wd
        if k == no_k - 1:
            # print('last')
            finish = len(fshot) - 1
        fig = plt.figure(figsize = (19, 17))
        plt.suptitle('Shot #' + str(f) + '\n' +
        
                     'K-statistic values: L:' + str(np.round(k_statistic[0],4)) +
                     ' D:' + str(np.round(k_statistic[1],4)) +
                     ' H:' + str(np.round(k_statistic[2],4)) +
                     ' Mean:' + str(np.round(k_statistic[3],4)),
                     fontsize = 'xx-large')
        shot_signals = get_raw_signals_in_window(fshot).swapaxes(0,1).astype(np.float32)
        fpad_max = np.max(shot_signals)
        fpad_min = np.min(shot_signals)
        times = fshot.time.values
        
        states_label = fshot.LHD_label.values
        ax_label = fig.add_subplot(8,1,(1,2))
        ax_label.grid()
        ax_label.set_title('Signal values (normalized) + labels, whole shot')
        ax_label.plot(times, shot_signals[:, 0], color='m', label='FIR')#, fshot.time.values, fshot.IP.values) #*1e19
        ax_label.plot(times, shot_signals[:, 1], color='g', label='DML')
        ax_label.plot(times, shot_signals[:, 2], color='b', label='PD')
        ax_label.plot(times, shot_signals[:, 3], color='c', label='IP')
        ax_label.set_xlabel('time, s')
        
        ax_label.axvspan(xmin = times[start], xmax = times[finish], color='k', alpha=.3)
        
        # pd_ip.set_title('PD signal and labels, whole shot')
        # pd_ip_lab.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, label')
        ax_label.fill_between(times, fpad_min, fpad_max,
                         where= states_label == 1, facecolor=cs[1], alpha=0.2, label='L, label')
        ax_label.fill_between(times, fpad_min, fpad_max,
                         where= states_label == 2, facecolor=cs[2], alpha=0.2, label='D, label')
        ax_label.fill_between(times, fpad_min, fpad_max,
                         where= states_label == 3, facecolor=cs[3], alpha=0.2, label='H, label')
        # pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
        #                  where= states_label == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        ax_label.legend(loc=2, prop={'size': leg_size}, ncol=8)
        
        
        
        
        ax_det = fig.add_subplot(8,1,(3,4))
        ax_det.grid()
        ax_det.set_title('Signal values (normalized) + seq2seq classification, whole shot')
        ax_det.plot(times, shot_signals[:, 0], color='m', label='FIR')#, fshot.time.values, fshot.IP.values) #*1e19
        ax_det.plot(times, shot_signals[:, 1], color='g', label='DML')
        ax_det.plot(times, shot_signals[:, 2], color='b', label='PD')
        ax_det.plot(times, shot_signals[:, 3], color='c', label='IP')
        ax_det.set_xlabel('time, s')
        # pd_ip_classifier.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        ax_det.axvspan(xmin = times[start], xmax = times[finish], color='k', alpha=.3)
        # pd_ip.set_title('PD signal and labels, whole shot')
        # pd_ip_classifier.vlines(x = fshot[fshot.ELM_prob > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, LSTM')
        ax_det.fill_between(times, fpad_min, fpad_max,
                         where= states_det == 1, facecolor=cs[1], alpha=0.2, label='Low, seq2seq')
        ax_det.fill_between(times, fpad_min, fpad_max,
                         where= states_det == 2, facecolor=cs[2], alpha=0.2, label='D, seq2seq')
        ax_det.fill_between(times, fpad_min, fpad_max,
                         where= states_det == 3, facecolor=cs[3], alpha=0.2, label='H, seq2seq')
        ax_det.legend(loc=2, prop={'size': leg_size}, ncol=8)
        
        fshot_zoom = fshot[start:finish]
        states_label_zoom = fshot_zoom.LHD_label.values
        signals_zoom = get_raw_signals_in_window(fshot_zoom).swapaxes(0,1).astype(np.float32)
        
        fpad_max = np.max(signals_zoom)
        fpad_min = np.min(signals_zoom)
        
        times_zoom = fshot_zoom.time.values
        
        # phd_max = np.max(fshot[start:finish].PD.values)
        ax_label_zoom = fig.add_subplot(8,1,(5,6))
        ax_label_zoom.set_title('Signal values (normalized) + labels, zoomed in')
        ax_label_zoom.plot(times_zoom, signals_zoom[:, 0], color='m', label='FIR')
        ax_label_zoom.plot(times_zoom, signals_zoom[:, 1], color='g', label='DML')
        ax_label_zoom.plot(times_zoom, signals_zoom[:, 2], color='b', label='PD')
        ax_label_zoom.plot(times_zoom, signals_zoom[:, 3], color='c', label='IP')
        
        ax_label_zoom.set_xlabel('time, s')
        # min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        ax_label_zoom.fill_between(times_zoom, fpad_min, fpad_max,
                         where= states_label_zoom == 1, facecolor=cs[1], alpha=0.2, label='Low, label')
        ax_label_zoom.fill_between(times_zoom, fpad_min, fpad_max,
                         where= states_label_zoom == 2, facecolor=cs[2], alpha=0.2, label='D, label')
        ax_label_zoom.fill_between(times_zoom, fpad_min, fpad_max,
                         where= states_label_zoom == 3, facecolor=cs[3], alpha=0.2, label='H, label')
        
        ax_label_zoom.set_ylim(.95*fpad_min, 1.05*fpad_max)
        ax_label_zoom.grid()
        ax_label_zoom.legend(loc=2, prop={'size': leg_size}, ncol=8)
        
        
        block_det_start = start // block_size
        block_det_end = finish // block_size
        block_det_zoom = states_blocks[block_det_start : block_det_end]
        # print(start, finish, block_det_start, block_det_end, block_det_zoom.shape)
        
        ax_det_zoom = fig.add_subplot(8,1,(7,8))
        ax_det_zoom.set_title('Signal values (normalized) + seq2seq classification, zoomed in')
        ax_det_zoom.plot(times_zoom, signals_zoom[:, 0], color='m', label='FIR')
        ax_det_zoom.plot(times_zoom, signals_zoom[:, 1], color='g', label='DML')
        ax_det_zoom.plot(times_zoom, signals_zoom[:, 2], color='b', label='PD')
        ax_det_zoom.plot(times_zoom, signals_zoom[:, 3], color='c', label='IP')
        
        ax_det_zoom.set_xlabel('time, s')
        
        
        for b in range(len(block_det_zoom)):
            block = int(block_det_zoom[b])
            block_times = times_zoom[b*block_size : b*block_size+block_size]
            # print(block_times)
            # ax_det_zoom.axvspan(block_times[0], block_times[-1], facecolor=cs[block], alpha=0.2, label=labels[block]) # , label=labels[block]
            ax_det_zoom.fill_between(block_times, fpad_min, fpad_max, facecolor=cs[block], alpha=0.2, label=labels[block])
        # ax_det_zoom.legend(loc=2, prop={'size': leg_size}, ncol=8)
        ax_det_zoom.grid()
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax_det_zoom.legend(by_label.values(), by_label.keys(), loc=2, prop={'size': leg_size}, ncol=8)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(test, format='pdf')
        plt.close('all')
        
        
        
    test.close()
    sys.stdout.flush()
    
def plot_shot_full(shot_id, fshot, directory, k_statistic):
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print 'start', start
        finish = (k+1) * wd
        if k == no_k - 1:
            # print('last')
            finish = len(fshot) - 1
        fig = plt.figure(figsize = (19, 17))
        plt.suptitle('Shot #' + str(f) + '\n' +

                     'K-statistic values: L:' + str(k_statistic[0]) +
                     ' D:' + str(k_statistic[1]) +
                     ' H:' + str(k_statistic[2]) +
                     ' Mean value:' + str(k_statistic[3]),
                     fontsize = 'xx-large')
        
        lab = fshot.LHD_label.values
        pd_ip_lab = fig.add_subplot(10,1,(1,2))
        pd_ip_lab.grid()
        pd_ip_lab.set_title('PD signal (normalized) + labels (majority opinion), (state and ELMs), whole shot')
        pd_ip_lab.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_lab.set_xlabel('time, s')
        
        pd_ip_lab.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_lab.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2, label='L, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2, label='D, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2, label='H, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        pd_ip_lab.legend(loc=2, prop={'size': leg_size}, ncol=2)
        
        pd_ip_classifier = fig.add_subplot(10,1,(3,4))
        pd_ip_classifier.grid()
        pd_ip_classifier.set_title('PD signal (normalized) + LSTM classification and ELM detection (threshold=.5), whole shot')
        pd_ip_classifier.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_classifier.set_xlabel('time, s')
        # pd_ip_classifier.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        pd_ip_classifier.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_classifier.vlines(x = fshot[fshot.ELM_prob > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, LSTM')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 1, facecolor=colors['green'], alpha=0.2, label='Low, LSTM')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 2, facecolor=colors['gold'], alpha=0.2, label='D, LSTM')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 3, facecolor=colors['red'], alpha=0.2, label='H, LSTM')
        pd_ip_classifier.legend(loc=2, prop={'size': leg_size})
        
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(10,1,(5,6))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values)
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        phd.legend(['PD'], loc=2, prop={'size': leg_size}, ncol=2)
        
        elm = fig.add_subplot(10,1,(7,8))
        elm.plot(fshot[start:finish].time.values, elms[start:finish], color = 'b', label='ELM, LSTM')
        fshot_cl = fshot[start:finish]
        fshot_cl_wh = np.where(fshot_cl.ELM_label.values == 1)[0]
        # print(fshot_cl_wh, len(fshot_cl_wh))
        temp = np.zeros(len(fshot_cl)).astype(bool)
        if len(fshot_cl_wh) > 0:
            for ind in fshot_cl_wh:
                # print(ind)
                ids = np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)
                ids = np.clip(ids, a_min=0, a_max=min(len(fshot), len(temp)-1))
                # ids = np.clip(ids, a_min=0, a_max=len(temp))
                # temp.extend(list(np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)))
                temp[ids] = True
        # temp = np.clip(list(set(temp)), a_min=0, a_max=len(fshot))
        # print(temp)
        # elm.vlines(x = fshot_cl[fshot_cl.sm_elm_label > .01].time.values, color = 'b', ymin = 0, ymax=1, alpha=0.2, label='ELM, label')
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= temp, facecolor = 'b', alpha=0.2, label='ELM, label')
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= fshot_cl.ELM_label.values == -1, facecolor = colors['lightgray'], alpha=0.8, label='No agreement, label')
        elm.set_xlabel('time, s')
        elm.set_title('elm labels (vertical lines) and direct LSTM elm output (continuous line)')
        # elm.legend(['ELM'], loc=2, prop={'size': leg_size}, ncol=2)
        elm.legend(loc=2, prop={'size': leg_size})
        elm.set_ylim([-0.1,1.1])
        elm.grid()
        
        state = fig.add_subplot(10,1,(9,10))
        state.plot(fshot[start:finish].time.values, states[start:finish,0], color = colors['green'], label='Low, LSTM') #L Mode
        
        state.plot(fshot[start:finish].time.values, states[start:finish,1], color = colors['gold'], label='Dither, LSTM') #d Mode
        state.plot(fshot[start:finish].time.values, states[start:finish,2], color = colors['red'], label='High, LSTM') #H Mode
        state.set_xlabel('time, s')
        state.set_title('state labels (solid color background, majority opinion) and direct LSTM outputs (colored lines)')
        # state.legend(['Low','Dither', 'High'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2, label='Low, label')
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, label')
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2, label='High, label')
        state.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_label.values == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        
        # state.legend(['low', 'dither', 'high'], loc=2)
        state.legend(loc=2, prop={'size': leg_size}) #,'H', 'D'
        state.set_ylim([-0.1,1.1])
        state.grid()
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(test, format='pdf')
        plt.close('all')
    test.close()
    sys.stdout.flush()
        
def plot_shot_simplified(shot_id, fshot, directory):
    # print('saving shot plot to ', directory)
    f = shot_id
    leg_size = 16
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    # states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    # fshot = normalize_signals_mean(fshot)
    # test = PdfPages(directory)
    
    fig = plt.figure(figsize = (15, 6))
    # plt.suptitle('TCV shot #' + str(f),
    #              fontsize =  30)
    
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 28})
    
    to_plot = ['PD']#, 'IP', 'DML', 'FIR']
    g_truth = fshot.LHD_label.values
    signals = np.asarray(fshot[to_plot].values)
    # print(signals.shape)
    # exit(0)
    
    pd_ip_classifier = fig.add_subplot(6,1,(1,4))
    lines = pd_ip_classifier.plot(fshot.time.values, signals, label='Data')#, fshot.time.values, fshot.IP.values) #*1e19 , label='PD, normalized'
    pd_ip_classifier.set_ylabel('Data values (norm.) ', labelpad = 0) #  Signal values \n (norm.)    PD values (norm.) 
    
    fpad_max = np.max(signals)
    fpad_min = np.min(signals)
    interval = fpad_max - fpad_min
    offset = interval / 5
    
    pd_ip_classifier.fill_between(fshot.time.values, fpad_min - offset, fpad_max + offset,
                     where= states_det == 1, facecolor=colors['green'], alpha=0.2, label='L')
    pd_ip_classifier.fill_between(fshot.time.values, fpad_min - offset, fpad_max + offset,
                     where= states_det == 2, facecolor=colors['gold'], alpha=0.2, label='D')
    pd_ip_classifier.fill_between(fshot.time.values, fpad_min - offset, fpad_max + offset,
                     where= states_det == 3, facecolor=colors['red'], alpha=0.2, label='H')
    pd_ip_classifier.get_xaxis().set_ticklabels([])
    # pd_ip_classifier.set_xlabel('t(s)')
    pd_ip_classifier.grid()
    leg1 = pd_ip_classifier.legend(loc=2, prop={'size': leg_size}, ncol=3)
    leg2 = pd_ip_classifier.legend(lines, to_plot, loc=3, ncol=len(lines), prop={'size': leg_size})
    plt.gca().add_artist(leg1)
    pd_ip_classifier_r = fig.add_subplot(6,1,(1,4), sharex=pd_ip_classifier, frameon=False)
    pd_ip_classifier_r.yaxis.set_label_position("right")
    pd_ip_classifier_r.set_ylabel('Prediction', rotation=270, labelpad = 40) #   Label    Prediction
    pd_ip_classifier_r.get_yaxis().set_ticklabels([])
    
    
    
    g_truth_plt = fig.add_subplot(6,1,(5,5))
    g_truth_plt.fill_between(fshot.time.values, 0, 1,
                     where= g_truth == 1, facecolor=colors['green'], alpha=0.2)
    g_truth_plt.fill_between(fshot.time.values, 0, 1,
                     where= g_truth == 2, facecolor=colors['gold'], alpha=0.2)
    g_truth_plt.fill_between(fshot.time.values, 0, 1,
                     where= g_truth == 3, facecolor=colors['red'], alpha=0.2,)
    g_truth_plt.fill_between(fshot.time.values, 0, 1,
                     where= g_truth == -1, facecolor=colors['gray'], alpha=1.0, label='No \n agreement')
    g_truth_plt.get_yaxis().set_ticklabels([])
    
    g_truth_plt_r = fig.add_subplot(6,1,(5,5), sharex=g_truth_plt, frameon=False)
    g_truth_plt_r.yaxis.set_label_position("right")
    g_truth_plt_r.set_ylabel('Gr. \n Truth', rotation=270, labelpad = 70)
    g_truth_plt_r.get_yaxis().set_ticklabels([])
    
    g_truth_plt.set_xlabel('t(s)')
    g_truth_plt.grid()
    
    # plt.savefig(directory)
    plt.show()
    # print('should be showing')
    # exit(0)
    # plt.close('all')
    
    # test.close()
    sys.stdout.flush()
    matplotlib.rcParams.update({'font.size': fs})
    # print(directory)
              
def plot_shot_cnn_viterbi_full(shot_id, fshot, directory, dice_cf, k_statistic):
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    # states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        finish = (k+1) * wd
        if k == no_k - 1:
            finish = len(fshot) - 1
        fig = plt.figure(figsize = (19, 17))
        plt.suptitle('Shot #' + str(f) + '\n' +
                     'Dice coefficient values: L:' + str(dice_cf[0])  +
                     ' D:' + str(dice_cf[1]) +
                     ' H:' + str(dice_cf[2]) +
                     ' Mean value:' + str(dice_cf[3]) + '\n' +
                     'K-statistic values: L:' + str(k_statistic[0]) +
                     ' D:' + str(k_statistic[1]) +
                     ' H:' + str(k_statistic[2]) +
                     ' Mean value:' + str(k_statistic[3]),
                     fontsize = 'xx-large')
        
        lab = fshot.LHD_label.values
        pd_ip_lab = fig.add_subplot(10,1,(1,2))
        pd_ip_lab.grid()
        pd_ip_lab.set_title('PD signal (normalized) + labels (majority opinion), (state and ELMs), whole shot')
        pd_ip_lab.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_lab.set_xlabel('time, s')
        
        pd_ip_lab.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_lab.vlines(x = fshot[fshot.ELM_label > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 1, facecolor=colors['green'], alpha=0.2, label='L, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 2, facecolor=colors['gold'], alpha=0.2, label='D, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == 3, facecolor=colors['red'], alpha=0.2, label='H, label')
        pd_ip_lab.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= lab == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        pd_ip_lab.legend(loc=2, prop={'size': leg_size}, ncol=2)
        
        pd_ip_classifier = fig.add_subplot(10,1,(3,4))
        pd_ip_classifier.grid()
        pd_ip_classifier.set_title('PD signal (normalized) + Convnet classification (obtained through threshold + state machine) and ELM detection (threshold=.5), whole shot')
        pd_ip_classifier.plot(fshot.time.values, fshot.PD.values, label='PD')#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd_ip_classifier.set_xlabel('time, s')
        # pd_ip_classifier.legend(['PD', 'IP'], loc=2, prop={'size': leg_size})
        pd_ip_classifier.axvspan(xmin = fshot.time.values[start], xmax = fshot.time.values[finish], color='k', alpha=.3)
        
        fpad_max = np.max(np.concatenate((fshot.PD.values, fshot.IP.values)))
        fpad_min = np.min(np.concatenate((fshot.PD.values, fshot.IP.values)))
        # pd_ip.set_title('PD signal and labels, whole shot')
        pd_ip_classifier.vlines(x = fshot[fshot.ELM_prob > .5].time.values, color = 'b', ymin = fpad_min, ymax=fpad_max, label='ELM, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 1, facecolor=colors['green'], alpha=0.2, label='Low, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 2, facecolor=colors['gold'], alpha=0.2, label='D, CNN')
        pd_ip_classifier.fill_between(fshot.time.values, fpad_min, fpad_max,
                         where= states_det == 3, facecolor=colors['red'], alpha=0.2, label='H, CNN')
        pd_ip_classifier.legend(loc=2, prop={'size': leg_size})
        
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(10,1,(5,6))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values)
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        phd.legend(['PD'], loc=2, prop={'size': leg_size}, ncol=2)
        
        elm = fig.add_subplot(10,1,(7,8))
        elm.plot(fshot[start:finish].time.values, elms[start:finish], color = 'b', label='ELM, CNN')
        fshot_cl = fshot[start:finish]
        fshot_cl_wh = np.where(fshot_cl.ELM_label.values == 1)[0]
        # print(fshot_cl_wh, len(fshot_cl_wh))
        temp = np.zeros(len(fshot_cl)).astype(bool)
        if len(fshot_cl_wh) > 0:
            for ind in fshot_cl_wh:
                # print(ind)
                ids = np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)
                ids = np.clip(ids, a_min=0, a_max=min(len(fshot), len(temp)-1))
                # ids = np.clip(ids, a_min=0, a_max=len(temp))
                # temp.extend(list(np.linspace(ind - 10, ind + 10, num=21, endpoint=True).astype(int)))
                temp[ids] = True
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= temp, facecolor = 'b', alpha=0.2, label='ELM, label')
        elm.fill_between(fshot_cl.time.values, 0., 1.,
                         where= fshot_cl.ELM_label.values == -1, facecolor = colors['lightgray'], alpha=0.8, label='No agreement, label')
        elm.set_xlabel('time, s')
        elm.set_title('elm labels (vertical lines) and direct CNN elm output (continuous line)')
        # elm.legend(['ELM'], loc=2, prop={'size': leg_size}, ncol=2)
        elm.legend(loc=2, prop={'size': leg_size})
        elm.set_ylim([-0.1,1.1])
        elm.grid()
        
        
        t_list = get_trans_ids()#+ ['no_trans']
        c = ['r','g','y',colors['teal'],'c','m',colors['black']]
        t_list = [t + '_det' for t in t_list]
        # transitions_det = fshot[t_list]
        trans = fig.add_subplot(10,1,(9,10))
        trans.set_xlabel('time, s')
        trans.set_title('direct CNN transition outputs (colored lines) and beam search output states (color background)')
        fshot_sl = fshot[start:finish]
        # print(fshot_sl.columns)
        for t_ind, t in enumerate(t_list):
            # print(c[t])
            # trans.plot(fshot[start:finish].time.values, transitions_det[start:finish][t_ind].values, color = c[t])
            trans.vlines(x = fshot_sl[fshot_sl[t].values == 1].time.values, color = c[t_ind], ymin = 0, ymax=1, alpha=1, linewidth=3, label=t[:2] + ', beam')
            
            trans.plot(fshot_sl.time.values, fshot_sl[t + '_prob'].values, color = c[t_ind])
        # trans.legend(get_trans_ids() + ['no_trans'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_det.values == 1, facecolor='g', alpha=0.2, label='Low, beam')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_det.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, beam')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_det.values == 3, facecolor='r', alpha=0.2, label='High, beam')
        trans.fill_between(fshot[start:finish].time.values, 0., 1.,
                         where= fshot[start:finish].LHD_det.values == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, beam')
        t_list = get_trans_ids()
        t_list = [t + '_lab' for t in t_list]
        # fshot_sl = fshot[start:finish]
        # for t, t_ind in enumerate(t_list):
        #     # print(fshot_sl[t_ind].shape)
        #     # print(fshot_sl[fshot_sl[t_ind].values > .5].time.shape)
        #     trans.vlines(x = fshot_sl[fshot_sl[t_ind].values > .01].time.values, color = c[t], ymin = 0, ymax=1, alpha=1, linewidth=3)
        trans.legend(loc=2, prop={'size': leg_size})
        trans.set_ylim([-0.1,1.1])
        trans.grid()
        
        
        
        plt.tight_layout(rect=[0, 0, 1, 0.92]) #rect=[0, 0, 1, 0.92])
        fig.savefig(test, format='pdf')
        plt.close('all')
        
        
    test.close()
    sys.stdout.flush()
        
def plot_shot_lstm_sig_lab(shot_id, fshot, directory):
    print('saving shot plot to ', directory)
    f = shot_id
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    elms = fshot.ELM_prob.values
    states = np.vstack((fshot.L_prob.values, fshot.D_prob.values, fshot.H_prob.values)).swapaxes(0,1)
    states_det = fshot.LHD_det.values
    # print('states', states.shape)

    fshot = normalize_signals_mean(fshot)
    test = PdfPages(directory)
    wd = 500
    leg_size = 16
    no_k = len(fshot)//wd + 1
    if len(fshot)%wd == 0:
        no_k -= 1
    for k in range(no_k):
        # print k
        start = int((k-.25) * wd)
        if k == 0:
            start = k * wd
        # print 'start', start
        finish = (k+1) * wd
        if k == no_k - 1:
            # print('last')
            finish = len(fshot) - 1
        fig = plt.figure(figsize = (19, 7))
        plt.suptitle('Shot #' + str(f) + '\n' ,
                     fontsize = 'xx-large')
        
        lab = fshot.LHD_label.values
        
        
        phd_max = np.max(fshot[start:finish].PD.values)
        phd = fig.add_subplot(4,1,(1,2))
        # phd.set_title('photodiode values + smoothed labels')
        phd.set_title('Photodiode (plasma emissivity) values, normalized')
        phd.plot(fshot[start:finish].time.values, fshot[start:finish].PD.values, label='PD')
        phd.set_xlabel('time, s')
        # trans = mtransforms.blended_transform_factory(phd.transData, phd.transAxes)
        min_d, max_d = np.min(fshot[start:finish].PD.values), np.max(fshot[start:finish].PD.values)
        
        phd.set_ylim(.95*min_d, 1.05*max_d)
        phd.grid()
        
        phd.fill_between(fshot[start:finish].time.values, min_d, max_d,
                         where= fshot[start:finish].LHD_label.values == 1, facecolor='g', alpha=0.2, label='Low, label')
        phd.fill_between(fshot[start:finish].time.values, min_d, max_d,
                         where= fshot[start:finish].LHD_label.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, label')
        phd.fill_between(fshot[start:finish].time.values, min_d,max_d,
                         where= fshot[start:finish].LHD_label.values == 3, facecolor='r', alpha=0.2, label='High, label')
        phd.fill_between(fshot[start:finish].time.values, min_d, max_d,
                         where= fshot[start:finish].LHD_label.values == -1, facecolor=colors['lightgray'], alpha=0.8, label='No agreement, label')
        phd.legend(loc=2, prop={'size': leg_size}, ncol=2)

        state = fig.add_subplot(4,1,(3,4))
        # state.plot(fshot[start:finish].time.values, states[start:finish,0], color = colors['green'], label='Low, LSTM') #L Mode
        # 
        # state.plot(fshot[start:finish].time.values, states[start:finish,1], color = colors['gold'], label='Dither, LSTM') #d Mode
        # state.plot(fshot[start:finish].time.values, states[start:finish,2], color = colors['red'], label='High, LSTM') #H Mode
        state.set_xlabel('time, s')
        state.set_title('state labels (solid color background, majority opinion) and direct LSTM outputs (colored lines)')
        # state.legend(['Low','Dither', 'High'], loc=2, prop={'size': leg_size}) #,'H', 'D'
        
        # ts = fshot[start:finish].time.values
        # ys = fshot[start:finish].LHD_det.values
        # le = len(ts)
        # ts2 = np.linspace(ts[0], ts[-1] + 1, le * 100)
        # a = np.zeros(len(ts))
        # b = np.ones(len(ts))
        # a2 = np.interp(ts2, ts, a)
        # b2 = np.interp(ts2, ts, b)
        # ys2 = np.interp(ts2, ts, ys)
        # ts, a, b, ys = ts2, a2, b2, ys2
        # 
        # state.fill_between(ts, a, b,
        #                  where= ys2 == 1, facecolor='g', alpha=0.2, label='Low, LSTM')
        # state.fill_between(ts, a, b,
        #                  where= ys2 == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, LSTM')
        # state.fill_between(ts, a, b,
        #                  where= ys2 == 3, facecolor='r', alpha=0.2, label='High, LSTM')
        
        
        state.fill_between(fshot[start:finish].time.values, min_d, max_d,
                         where= fshot[start:finish].LHD_det.values == 1, facecolor='g', alpha=0.2, label='Low, LSTM')
        state.fill_between(fshot[start:finish].time.values, min_d, max_d,
                         where= fshot[start:finish].LHD_det.values == 2, facecolor=colors['gold'], alpha=0.2, label='Dither, LSTM')
        state.fill_between(fshot[start:finish].time.values, min_d,max_d,
                         where= fshot[start:finish].LHD_det.values == 3, facecolor='r', alpha=0.2, label='High, LSTM')
        
        # state.legend(['low', 'dither', 'high'], loc=2)
        state.legend(loc=2, prop={'size': leg_size}) #,'H', 'D'
        state.set_ylim([-0.1,1.1])
        state.grid()
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(test, format='pdf')
        plt.close('all')
    test.close()
    sys.stdout.flush()
               
def plot_attention_prediction(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot, convolutional_stride,
                              conv_w_size, block_size, fname, max_source_sentence_chars,
                              max_train_target_words, look_ahead, num_channels):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    blocks = trans_detected
    # blocks = np.arange(11)
    # blocks_in_subseq = timesteps // block_size
    # encoder_inputs_to_blocks = np.empty(len(blocks), timesteps, num_channels)
    blocks_per_source_sentence = max_train_target_words
    target_chars_per_sentence = blocks_per_source_sentence * block_size
    remainder = max_source_sentence_chars - target_chars_per_sentence- 2*look_ahead
    
    handler = PdfPages(fname)
    # print(blocks_per_source_sentence, max_source_sentence_chars)
    # print(attention_weights_sequence.shape)
    # exit(0)
    cumul=0
    for ind, block_ind in enumerate(blocks):
        # signal_start_ind = block_ind * block_size - remainder
        # signal_end_ind = signal_start_ind + max_source_sentence_chars
        subseqs_until_block = block_ind // blocks_per_source_sentence
        signal_start_ind = (subseqs_until_block) * max_source_sentence_chars -2*look_ahead*subseqs_until_block - remainder*subseqs_until_block
        signal_end_ind = signal_start_ind + max_source_sentence_chars
        # print(block_ind, block_ind // blocks_per_source_sentence, signal_start_ind, signal_end_ind, len(shot_signals))
        if signal_end_ind > len(shot_signals):
            print('warning, you are trying to plot a block outside the range specified in the call to the main function.')
            handler.close()
            sys.stdout.flush()
            return
            # exit(0)
        shot_signals_sliced = shot_signals[signal_start_ind:signal_end_ind]
        times_sliced = times[signal_start_ind:signal_end_ind]
        
        # block_start_ind = block_ind
        # block_end_ind = block_start_ind + blocks_per_source_sentence
        # decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
        # attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
        
        block_start_ind = subseqs_until_block * blocks_per_source_sentence
        block_end_ind = block_start_ind + blocks_per_source_sentence
        decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
        attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
        
        plt.rc('font', **font)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
        cs = ['g',colors['gold'],'r']
        fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)
    
        p1 = fig.add_subplot(2,1,1)
        p1.set_ylabel('signal values')
        # p1.plot(times,pd, label='PD')
        clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
        
    
        p1.plot(times_sliced, shot_signals_sliced[:, 0], color='m', label='FIR')
        p1.plot(times_sliced, shot_signals_sliced[:, 1], color='g', label='DML')
        p1.plot(times_sliced, shot_signals_sliced[:, 2], color='b', label='PD')
        p1.plot(times_sliced, shot_signals_sliced[:, 3], color='c', label='IP')
            # frac = (w%4)/4
            # p1.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=clist[w%4], alpha=.7, ymin = frac, ymax = frac+.25)
        p1.set_title('Encoder Input (windowed signal values)' + str(shot_signals_sliced.shape))
        p1.grid(zorder=0)
        p1.set_xlim([times_sliced[0], times_sliced[-1]])
        # p1.set_ylabel('PD (norm.)')
        # p1.xaxis.set_ticklabels(times_to_plot)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # p1.legend(loc=2, prop={'size': 22})
        plt.legend(by_label.values(), by_label.keys(), loc=2)
        
        p2 = fig.add_subplot(2,1,2)
        p2.set_title('Decoder Output (sequence of transition blocks)' + str(decoded_sequence_sliced.shape))
        p2.set_ylabel('P(transition)')
        p2.set_xlim([times_sliced[0], times_sliced[-1]])
        labels = get_trans_ids()+['no_trans']
        labels = ['L', 'D', 'H']
        
        times_sliced_look_ahead = times_sliced[look_ahead : - look_ahead - remainder]
        # print(times_sliced_look_ahead[0], times_sliced_look_ahead[-1])
        
        attn_weights = convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced)
        # attn_weights = attention_weights_seq_sliced
        # print(attn_weights_convolved.shape)
        # exit(0)
        # ylims = p1.get_ylim()
        for k in range(len(decoded_sequence_sliced)):
            times_block_sliced = times_sliced_look_ahead[k*block_size : k*block_size+block_size]
            # print(k, times_block_sliced[0], times_block_sliced[-1])
            # times_block_sliced = times_sliced[k*block_size : k*block_size+block_size]
            # print(times_to_plot)
            block = decoded_sequence_sliced[k] - 1
            # block = 6
            # print(block.shape, block)
            # to_color = np.where(block==1)[0]#[0]
            # print(to_color, len(to_color))
            # if len(to_color) > 0:
                # c_ind = int(to_color[0])
            
            # if(block_ind % len(decoded_sequence_sliced)) == k:
            # highest_att_ids = np.argsort(attention_weights_seq_sliced[k])[-5:]
            # print(highest_att_ids)
            if k == block_ind % blocks_per_source_sentence:
                # print(k, block_ind, block_size, len(decoded_sequence_sliced))
                alpha = 1
                # positions = np.arange(19)
                # positions *= 10
                # positions += 10
                # print(attention_weights_seq_sliced.shape)
                # p1.set_xticks(positions)
                # p1.set_xticklabels(np.round(attention_weights_seq_sliced[:,0], 3))
                # print(np.argsort(attention_weights_seq_sliced, axis=0)) #3-highest probability
                # print(attention_weights_seq_sliced[k].shape)
                sorted_att_ids = np.argsort(attn_weights[k], axis=0) #lowest to highest
                # highest_att_ids = np.asarray([0,1])
                # print(highest_att_ids)
                max_alpha = .5
                alphas = np.arange(0, max_alpha, max_alpha/attn_weights.shape[1])[::-1]
                alphas = attn_weights[k]
                alphas -= np.min(alphas)
                distortion_factor = max_alpha / np.max(alphas)
                alphas *= distortion_factor
                # plt.plot(alphas)
                alphas = np.clip(alphas, a_min=0, a_max=max_alpha)
                alphas = max_alpha - alphas
                # print(alphas)
                for att_id_ind, att_id in enumerate(sorted_att_ids):
                    att_time_st = att_id * convolutional_stride
                    att_time_end = att_time_st + convolutional_stride 
                    # att_time_end = att_time_st + conv_w_size - 1
                    #careful with rounding errors in next line
                    if times_sliced[att_time_st] == times_sliced[-convolutional_stride]:
                        att_time_end -= 1
                    left_limit, right_limit = times_sliced[att_time_st], times_sliced[att_time_end]    
                    # time_int = times_sliced[att_time_st : att_time_end]
                    p1.axvspan(left_limit, right_limit, facecolor='black', alpha=alphas[att_id], label=att_id_ind + 1)
                    # p1.fill_between(times_sliced, ylims[0], ylims[1], where=times_sliced, facecolor='black', alpha=alphas[att_id_ind])
                    
                    # p1.text(times_sliced[att_time_st+conv_w_size//2-1], 1.5, np.round(attention_weights_seq_sliced[att_id, 0],3)[0], rotation=90) 
            else:
                alpha = .3
                
            # if block != 6:
            #     p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
            # else:
            #     p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor='gray', alpha=alpha, label='No trans')
            # print(block)
            # print(times_block_sliced[0])
            # print(times_block_sliced[-1])
            p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
        # exit(0)
        p2.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
       
        plt.tight_layout()
        # plt.show()
        # plt.savefig(fname)
        fig.savefig(handler, format='pdf')
        plt.close('all')
    handler.close()
    sys.stdout.flush()

def convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced):
    attn_weights_convolved = []
    windows_per_conv = conv_w_size//convolutional_stride
    num_windows = attention_weights_seq_sliced.shape[1]
    for i in range(num_windows + windows_per_conv - 1):
        inds = np.arange(i - windows_per_conv +1, i + 1)
        inds = inds[np.logical_and(inds>= 0, inds <= num_windows - 1)]
        vals = attention_weights_seq_sliced[:, inds]
        attn_weights_convolved.append(np.sum(vals, axis=1)/vals.shape[1])
    attn_weights_convolved = np.asarray(attn_weights_convolved).swapaxes(0,1)
    return attn_weights_convolved

def plot_attention_matrix(shot_signals, decoded_sequence, label_sequence, trans_detected, attention_weights_sequence, times, shot, convolutional_stride,
                          conv_w_size, block_size, fname, max_source_sentence_chars,
                          max_train_target_words, look_ahead, num_channels):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    # blocks = trans_detected
    # blocks = np.arange(22)  
    # blocks_per_source_sentence = (max_source_sentence_chars - 2*look_ahead) // block_size
    # assert blocks_per_source_sentence == target_blocks_per_source_sentence
    blocks_per_source_sentence = max_train_target_words
    target_chars_per_sentence = blocks_per_source_sentence * block_size
    remainder = max_source_sentence_chars - target_chars_per_sentence- 2*look_ahead
    
    handler = PdfPages(fname)
    blocks = np.arange(len(decoded_sequence))
    
    # assert shot_signals.shape[0] >= decoded_sequence.shape[0]*block_size  + 2*look_ahead
    # print('in plot_attention_matrix')
    # print(blocks.shape)
    # print(attention_weights_sequence[:,:,0].shape)
    # print(attention_weights_sequence[0,:,0])
    # print(attention_weights_sequence[1,:,0])
    # print(attention_weights_sequence[18,:,0])
    # print(attention_weights_sequence[19,:,0])
    # # print(np.sum(attention_weights_sequence[:,:,0], axis=1))
    # # print(np.sum(attention_weights_sequence[:,:,0], axis=1)[:18])
    # # print(np.sum(attention_weights_sequence[:,:,0], axis=0).shape)
    # exit(0)
    for ind, block_ind in enumerate(blocks[::blocks_per_source_sentence]):
        # print(ind)
        # signal_start_ind = (block_ind // blocks_per_source_sentence) * max_source_sentence_chars - cumul
        signal_start_ind = (block_ind // blocks_per_source_sentence) * max_source_sentence_chars -2*look_ahead*ind - remainder*ind
        signal_end_ind = signal_start_ind + max_source_sentence_chars
        
        # print(signal_start_ind, signal_end_ind)
        if signal_end_ind > len(shot_signals):
            print('warning, you are trying to plot a block outside the range specified in the call to the main function.')
            handler.close()
            sys.stdout.flush()
            return
            # exit(0)
        shot_signals_sliced = shot_signals[signal_start_ind:signal_end_ind]
        times_sliced = times[signal_start_ind:signal_end_ind]
        # print(times[signal_start_ind:signal_end_ind])
        # exit(0)
        block_start_ind = (block_ind // blocks_per_source_sentence) * blocks_per_source_sentence
        block_end_ind = block_start_ind + blocks_per_source_sentence
        decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
        attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
        # print(attention_weights_sequence.shape, attention_weights_seq_sliced.shape, decoded_sequence_sliced.shape)
        # exit(0)
        plt.rc('font', **font)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
        cs = ['g','y','r']
        fig = plt.figure(figsize = (17, 8)) #figsize = (19, 5)
        # fig = plt.figure()
        gs1 = GridSpec(12, 19)
        ax1 = fig.add_subplot(gs1[0:3, 7:])
        ax2 = fig.add_subplot(gs1[3:, 7:])
        ax3 = fig.add_subplot(gs1[3:, 3:6])
        ax4 = fig.add_subplot(gs1[3:, 0:3])
        
        # p1 = fig.add_subplot(2,1,1)
        ax1.set_ylabel('Signal\n values')
        # p1.plot(times,pd, label='PD')
        clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
        
        # print(times_sliced, times_sliced.shape, shot_signals_sliced.shape, shot_signals.shape, times.shape)
    
        ax1.plot(times_sliced, shot_signals_sliced[:, 0], color='m', label='FIR')
        ax1.plot(times_sliced, shot_signals_sliced[:, 1], color='g', label='DML')
        ax1.plot(times_sliced, shot_signals_sliced[:, 2], color='b', label='PD')
        ax1.plot(times_sliced, shot_signals_sliced[:, 3], color='c', label='IP')
        ax1.xaxis.set_ticks_position('top')
        ax1.axvspan(times_sliced[0], times_sliced[look_ahead], facecolor='k', alpha=.2)
        ax1.axvspan(times_sliced[-look_ahead-remainder], times_sliced[-1], facecolor='k', alpha=.2)
        # ax1.set_xticklabels([])
        # plt.show()
            # frac = (w%4)/4
            # p1.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=clist[w%4], alpha=.7, ymin = frac, ymax = frac+.25)
        ax1.set_title('Encoder Input (signal values)' + str(shot_signals_sliced.shape) + '. Window size:' + str(conv_w_size) + '. Stride:' + str(convolutional_stride))
        ax1.set_xlim(times_sliced[0], times_sliced[-1])
        ax1.grid()
        # plt.show()
        # p1.set_ylabel('PD (norm.)')
        # p1.xaxis.set_ticklabels(times_to_plot)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # p1.legend(loc=2, prop={'size': 22})
        plt.legend(by_label.values(), by_label.keys(), loc=2)
        
        # p2 = fig.add_subplot(2,1,2)
        
        # ax3.set_title('Decoder Output (sequence of transition blocks)' + str(decoded_sequence.shape))
        
        ax3.set_xlabel('P(transition)')
        # ax3.set_ylabel('Decoder Output (blocks)')
        ax3.set_title('Decoder Output')
        
        
        labels = ['L', 'D', 'H']
        times_sliced_look_ahead = times_sliced[look_ahead : - look_ahead - remainder]
        
        for k in range(len(decoded_sequence_sliced)):
            times_block_sliced = times_sliced_look_ahead[k*block_size : k*block_size+block_size]
            # block = decoded_sequence_sliced[k, 0]
            block = decoded_sequence_sliced[k] - 1
            # print('block', block)
            # exit(0)
            alpha = .5
            ax3.axhspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
                  
        ax3.set_ylim(times_sliced_look_ahead[-1], times_sliced_look_ahead[0])
        ax3.set_yticks([])
        # for minor ticks
        ax3.set_yticks([], minor=True)
            # else:
        # exit(0)
        # ax3.text(-10, 0, 'Decoder Output (sequence of transition blocks)' + str(decoded_sequence.shape), rotation=90 )
        ax3.grid()
        # ax3.set_yticklabels([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc=1)
        
        # ax4.set_xlabel('P(transition)')
        ax4.set_xticks([])
        # for minor ticks
        ax4.set_xticks([], minor=True)
        ax4.set_ylabel('t(s)')
        ax4.set_title('Labels')
        
        labels_sequence_sliced = label_sequence[signal_start_ind + look_ahead :signal_end_ind - look_ahead - remainder]
        # print(label_sequence.shape, shot_signals.shape)
        # ax4.axhspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
        ax4.fill_betweenx(times_sliced_look_ahead, 0, 1,
                     where= labels_sequence_sliced == 1, facecolor=cs[0], alpha=0.5, label='L')
        ax4.fill_betweenx(times_sliced_look_ahead, 0, 1,
                     where= labels_sequence_sliced == 2, facecolor=cs[1], alpha=0.5, label='D')
        ax4.fill_betweenx(times_sliced_look_ahead, 0, 1,
                     where= labels_sequence_sliced == 3, facecolor=cs[2], alpha=0.5, label='H')
        ax4.set_ylim(times_sliced_look_ahead[-1], times_sliced_look_ahead[0])
        ax4.grid()
        ax4.legend()
        attn_weights_convolved = convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced)
        
        # mat_dim_col = num_windows + windows_per_conv - 1
        # attn_weights_mat = np.zeros(attention_weights_seq_sliced.shape[0], mat_dim_col)
        # for row in attention_weights_seq_sliced.shape[0]:
        #     for col in np.arange(mat_dim_col):
        #         attn_weights_mat[row, ]
        #     
        
        ax2.matshow(attn_weights_convolved, aspect='auto', vmin=0, vmax=1,)
        
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('right')
        
        ax2.set_xticks(np.arange(0, attn_weights_convolved.shape[1], 1))
        ax2.set_xticklabels(np.arange(1, attn_weights_convolved.shape[1] + 1, 1), rotation='vertical')
        ax2.set_yticks(np.arange(0, attn_weights_convolved.shape[0], 1))
        ax2.set_yticklabels(np.arange(1, attn_weights_convolved.shape[0] + 1, 1))
        ax2.set_xticks(np.arange(-.5, attn_weights_convolved.shape[1], 1), minor=True)
        ax2.set_yticks(np.arange(-.5, attn_weights_convolved.shape[0], 1), minor=True)
        # ax2.grid(which='minor')
        plt.tight_layout()
        # plt.show()
        # plt.savefig(fname)
        fig.savefig(handler, format='pdf')
        plt.close('all')
    handler.close()
    sys.stdout.flush()    
    
def plot_conf_mat(matrix, matrix_order, fname):
    handler = PdfPages(fname)
    fig = plt.figure()
    labels_sum = np.sum(matrix, axis=matrix_order[1])
    predictions_sum = np.sum(matrix, axis=matrix_order[0])
    ax = fig.add_subplot(111)
    # print(matrix)
    # print(labels_sum)
    norm_mat = np.round(matrix/labels_sum[:, np.newaxis], 4)
    # print(norm_mat)
    # cmap = 
    cmap = ListedColormap(sns.cubehelix_palette(start=.5, rot=-.5, dark=0.2, light=.66).as_hex())
    
    ax.matshow(norm_mat, cmap=cmap)
    axes_labels=['predictions', 'labels']
    # ax2=ax.twinx()
    labels = ['', 'L', 'D', 'H']
    
    # ax.set_xlabel(axes_labels[matrix_order[0]])
    
    ax.xaxis.set_label_position('top')
    
    
    # print(labels_sum, predictions_sum)
    # ax2.set_xlabel(labels_sum)
    ax.set_xlabel(axes_labels[matrix_order[0]])
    ax.set_ylabel(axes_labels[matrix_order[1]])
    
    for i in range(norm_mat.shape[0]):
        for j in range(norm_mat.shape[1]):
            text = ax.text(j, i, np.round(norm_mat[i, j], 4), ha="center", va="center", color="w")
    
    # ax2.set_yticks(labels_sum)
    # ax2.set_yticklabels(labels_sum)
    # tics = ax.get_yticks()
    # print(tics)
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    fig.savefig(handler, format='pdf')
    plt.close('all')
    handler.close()
    
    
    
if __name__ == "__main__":
    main()
