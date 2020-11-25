import matplotlib.pyplot as plt
from helper_funcs import get_roc_best
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from collections import OrderedDict
import csv

def plot_roc_curve(roc_curve, thresholds, roc_fname, title):
    youden_indexes, threshold, dist = get_roc_best(roc_curve) #best threshold for elms
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 24})
    f = PdfPages(roc_fname)
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111)
    fprs = []
    tprs = []
    for t in thresholds:
        fpr, tpr = roc_curve[t]
        fprs += [fpr]
        tprs += [tpr]
        # if t in (0, 1, threshold):
        # # if t in (threshold,):
        #     ax.annotate(t, (fpr, tpr))
        if t == 0:
            ax.annotate(t, (fpr-.1, tpr-.1))
        if t == 1:
            ax.annotate(t, (fpr, tpr))    
        if t == threshold:
            ax.annotate(t, (fpr, tpr-.1))        
        # p.scatter(fpr, tpr, alpha=1.0, c='blue', edgecolors='none')
    ax.plot(fprs, tprs, 'o-')
    # ax.set_xlim([-.005,.05])
    ax.set_xlim([-.11,1.1])
    ax.set_ylim([-.11,1.1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_yticks(ticks=[0, 0.5, 1.])
    ax.set_xticks(ticks=[0, 0.5, 1.])
    print('elm roc_curve', roc_curve)
    print('youden_indexes', youden_indexes)
    print('best roc threshold', threshold)
    fig.suptitle(title)
    
    matplotlib.rcParams.update({'font.size': fs})
    plt.tight_layout(rect=[0, 0, 1, .95])
    
    fig.savefig(f, format='pdf')
    f.close()
    
def plot_kappa_histogram(k_indexes, histo_fname, title):
    fs = matplotlib.rcParams['font.size']
    bs = [0., 0.2, 0.4, 0.6, 0.8, 1]
    st_and_mean = ['Low', 'Dither', 'High', 'Mean']
    f = PdfPages(histo_fname)
    fig = plt.figure(figsize = (6, 5))
    
    print('------------k_indices:------------')
    for k, state in enumerate(st_and_mean):
        k_ind = k_indexes[:, k]
        matplotlib.rcParams.update({'font.size': 16})
        p = fig.add_subplot(2,2,k+1)
        p.hist(k_ind, bins=5, range=(0, 1))
        # sns.distplot(k_ind, bins=5, kde=False, hist_kws={'range': (0.0, 1.0)})
        # p.xaxis.set_ticklabels([-.2,] + bs.append(1.2), rotation=45)
        p.set_title(state)
        p.set_ylim(bottom = None, top=k_ind.shape[0] + 1)
        # p.set_xlim([-.1, 1.1])
        fig.text(.02, .5 + 0.025, 'Frequency', ha='center', va='center', rotation='vertical')
        fig.text(.5 + 0.025, .02, 'Score', ha='center', va='center', rotation='horizontal')
        # fig.suptitle('K-index Scores')
        fig.suptitle(title)
        matplotlib.rcParams.update({'font.size': fs})
        # plt.tight_layout()
        fig.tight_layout(rect=[0.025, 0.025, 1, .95])
        p.set_xticks(bs)
        p.tick_params(axis='x', rotation=45)
        if k < 2:
            p.xaxis.set_ticklabels([])
        if k %2 == 1:
            p.yaxis.set_ticklabels([])
        hist, bin_edges = np.histogram(k_ind, bins=bs)
        print('histogram for', state, hist) #k_ind
    fig.savefig(f, format='pdf')
    f.close()
    
def plot_dice_histogram(dice_cfs, histo_fname):
    fs = matplotlib.rcParams['font.size']
    bs = [0., 0.2, 0.4, 0.6, 0.8, 1]
    st_and_mean = ['Low', 'Dither', 'High', 'Mean']
    f = PdfPages(histo_fname)
    fig = plt.figure(figsize = (6, 5))
    print('------------dices:------------')
    for k, state in enumerate(st_and_mean):
        dice_state = dice_cfs[:, k]
        matplotlib.rcParams.update({'font.size': 16})
        p = fig.add_subplot(2,2,k+1)
        p.hist(dice_state, bins=5, range=(0, 1))
        p.set_title(state)
        p.set_ylim(bottom = None, top=dice_state.shape[0] + 1)
        
        fig.text(.02, .5 + 0.025, 'Frequency', ha='center', va='center', rotation='vertical')
        fig.text(.5 + 0.025, .02, 'Score', ha='center', va='center', rotation='horizontal')
        # fig.suptitle('Dice Coefficient Scores, using best threshold for transitions')
        # plt.tight_layout()
        matplotlib.rcParams.update({'font.size': fs})
        p.set_xticks(bs)
        p.tick_params(axis='x', rotation=45)
        if k < 2:
            p.xaxis.set_ticklabels([])
        if k %2 == 1:
            p.yaxis.set_ticklabels([])
        fig.tight_layout(rect=[0.025, 0.025, 1, 1])
        hist, bin_edges = np.histogram(dice_state, bins=bs)
        print('histogram for', state, hist, dice_state)
    fig.savefig(f, format='pdf')
    f.close()
    
def out_sorted_scores(k_indexes_dic, fpath):
    st_and_mean = ['Low', 'Dither', 'High', 'Mean']
    for k, state in enumerate(st_and_mean):
        path = fpath + state + '.csv'
        # print('Shots ordered by lowest to highest k-index, sorted by ' + state)
        dd = OrderedDict(sorted(k_indexes_dic.items(), key=lambda x: x[1][k]))
        # print(dd)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['shot', 'Low DC', 'Dither DC', 'High DC', 'Mean DC'])
            w.writeheader()
            for key, val in dd.items():
                row = {'shot': key}
                for c_l, l in enumerate(st_and_mean):
                    row[l + ' DC'] = val[c_l]
                w.writerow(row)