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
import matplotlib as mplt

def plot_shot_versions(shot_t, shot_sig, fshots_states, fshots_elms, save_dir='', shot_id='', labelers=[]):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    len_sh = len(shot_t)
    no_shs = len(fshots_states)
    leg_size = 16
    fig = plt.figure(figsize = (25, 14))
    # fig = plt.figure()
    # fig.suptitle('#' + shot_id)
    test = PdfPages(save_dir + shot_id + '.pdf')
    mplt.rcParams.update({'font.size': 22})
    for k in range(no_shs):
        # print('first one')
        
        labeler = labelers[k]
        print('labeler', labeler)
        states = fshots_states[k]
        # print(len(states), len_sh)
        assert len(states) == len_sh
        elms = fshots_elms[k]
        # print('states', states.shape)
        sid = shot_id
        # print(2*no_shs,1,(2*k+1, 2*k+2))
        pd = fig.add_subplot(2*no_shs,1,(2*k+1, 2*k+2))
        pd.grid()
        if labeler == 'lstm':
            pd.set_title('#' + shot_id + ', classified by ' + labeler)
        else:
            pd.set_title('#' + shot_id + ', labeled by ' + labeler)
        pd.plot(shot_t, shot_sig)#, fshot.time.values, fshot.IP.values) #*1e19
        # amax = np.argmax(fshot., axis=1)
        # power_and_ip.plot(fshot.time.values, )
        pd.set_xlabel('time, s')
        fig.text(0.0, 0.5, 'PD signal (normalized)', va='center', rotation='vertical')
        # pd.set_ylabel('PD signal (normalized)')
        fpad_max = np.max(shot_sig)
        fpad_min = np.min(shot_sig)
        pd.fill_between(shot_t, fpad_min, fpad_max,
                         where= states == 1, facecolor=colors['green'], alpha=0.2)
        pd.fill_between(shot_t, fpad_min, fpad_max,
                         where= states == 2, facecolor=colors['gold'], alpha=0.2)
        pd.fill_between(shot_t, fpad_min, fpad_max,
                         where= states == 3, facecolor=colors['red'], alpha=0.2)
        
        pd.legend(['PD', 'Low', 'Dither', 'High'], loc=2, prop={'size': leg_size})    
            
         
    plt.tight_layout()
    fig.savefig(test, format='pdf')
    plt.close('all')
    test.close()
    sys.stdout.flush()


# def plot_shot_versions(shot_t, fshot, save_dir):
#     colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#     len_sh = len(shot_t)
#     no_shs = len(fshots_states)
#     leg_size = 16
#     fig = plt.figure(figsize = (25, 14))
#     # fig = plt.figure()
#     # fig.suptitle('#' + shot_id)
#     test = PdfPages(save_dir + shot_id + '.pdf')
#     mplt.rcParams.update({'font.size': 22})
#     for k in range(no_shs):
#         labeler = labelers[k]
#         print('labeler', labeler)
#         states = fshots_states[k]
#         # print(len(states), len_sh)
#         assert len(states) == len_sh
#         elms = fshots_elms[k]
#         # print('states', states.shape)
#         sid = shot_id
#         # print(2*no_shs,1,(2*k+1, 2*k+2))
#         pd = fig.add_subplot(2*no_shs,1,(2*k+1, 2*k+2))
#         pd.grid()
#         if labeler == 'lstm':
#             pd.set_title('#' + shot_id + ', classified by ' + labeler)
#         else:
#             pd.set_title('#' + shot_id + ', labeled by ' + labeler)
#         pd.plot(shot_t, shot_sig)#, fshot.time.values, fshot.IP.values) #*1e19
#         # amax = np.argmax(fshot., axis=1)
#         # power_and_ip.plot(fshot.time.values, )
#         pd.set_xlabel('time, s')
#         fig.text(0.0, 0.5, 'PD signal (normalized)', va='center', rotation='vertical')
#         # pd.set_ylabel('PD signal (normalized)')
#         fpad_max = np.max(shot_sig)
#         fpad_min = np.min(shot_sig)
#         pd.fill_between(shot_t, fpad_min, fpad_max,
#                          where= states == 1, facecolor=colors['green'], alpha=0.2)
#         pd.fill_between(shot_t, fpad_min, fpad_max,
#                          where= states == 2, facecolor=colors['gold'], alpha=0.2)
#         pd.fill_between(shot_t, fpad_min, fpad_max,
#                          where= states == 3, facecolor=colors['red'], alpha=0.2)
#         
#         pd.legend(['PD', 'Low', 'Dither', 'High'], loc=2, prop={'size': leg_size})    
#             
#          
#     plt.tight_layout()
#     fig.savefig(test, format='pdf')
#     plt.close('all')
#     test.close()
#     sys.stdout.flush()
        
if __name__ == "__main__":
    main()
