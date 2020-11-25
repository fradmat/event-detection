import matplotlib.pyplot as plt
import numpy as np

def main():
    federico_test_scores = np.asarray([2,6,4,3,3,3,8,9,8,7,7,9,5,7,5,7,9,8,7,9,5,7,6,7,8,8,7,7,8,8,9,8,5,7,9,9,8,8,8,8,9,7,0,0,0,0]) / 10 ##overall shot score, ordered by dither dc score
    federico_train_scores = np.asarray([9,9,9,9,10,9,9,10,9,8]) / 10    ##overall shot score, ordered by dither dc score
    # print(federico_test_scores)
    print(federico_train_scores)
    dice_train_scores = np.asarray([0.97560799770537,0.9885673156660101,0.9453046044648153,0.9973979423392043,
                                    0.991266944833146,0.9644913270520126,0.9941567556468179,0.9894663083023153,0.9891423547207646,0.9867805293345088]) ##overall shot score, ordered by dither dc score
    dice_test_scores = np.asarray([0.3567100305256557,0.9963048514429058,0.9359238530302133,0.5416099180631606,0.7782785380975459,0.8198925309734217,
                                   0.9355637299077082,0.9892242790260868,0.9396247484671874,0.8353931941221331,0.9462748918758063,0.9808842388714871,
                                   0.6290731266331198,0.7865819133213826,0.846448075178697,0.5906199484817758,0.7743814191026878,0.6900383370513363,
                                   0.9957218469861245,0.9748445402455972,0.9399105878757367,0.8596735785542253,0.8776631869560156,0.9545188219472662,
                                   0.9590433866706826,0.8658144132587803,0.9402143734316648,0.5025387836234777,0.9706147399714536,0.9277728331693885,
                                   0.9734810915536445,0.9177063917790214,0.6180097260422971,0.906655001821921,0.9308375845990914,0.991617080698324,
                                   0.9758058263962789,0.9770396955246364,0.9840802530710628,0.9835186161405296,0.9925079243800617,0.9054559662078945,
                                   0.9996359954877115,0.9291455458128962,0.9085475369928476,0.7315324757953166])
    train_shots_dither_dice_ordered = [31211,32794,32592,30262,58182,47962,32191,32716,32195,32911]
    test_shots_dither_dice_ordered = [61057,26386,57751,31650,33446,43454,31807,52302,30290,61021,33638,31718,60275,45105,57103,60097,31554,42062,
                                      30268,30225,33459,45103,57218,34010,53601,30043,29511,39872,34309,33942,30310,59825,33281,33271,31839,48580,
                                      33188,33567,58460,62744,56662,42197,26383,42514,30044,30197]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    
    # for data, color, group in zip(data, colors, groups):
    # x, y = data
    ax.set_xlim([-.1,1.1])
    ax.set_ylim([-.1,1.1])
    ax.set_xlabel('Federico\'s Score (divided by 10)')
    ax.set_ylabel('Dice Score (Weighted average for all states)')
    # groups = ("coffee", "tea", "water")
    # ax.scatter(federico_train_scores, dice_train_scores, alpha=0.8, c='blue', edgecolors='none', label='train')
    # ax.scatter(federico_test_scores, dice_test_scores, alpha=0.8, c='red', edgecolors='none', label='test')
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    mail_to_federico = (57751,43454,33446,33459,57103,26386,57218,30268,34010,26383,42514,30044,30197)
    for i,s in enumerate(train_shots_dither_dice_ordered):
        f_sc, d_sc = federico_train_scores[i], dice_train_scores[i]
        if s in mail_to_federico:
            ax.scatter(f_sc, d_sc, alpha=1.0, c='blue', edgecolors='none') #,'label' = train
        else:
            ax.scatter(f_sc, d_sc, alpha=.1, c='blue', edgecolors='none')
        
        ax.text(f_sc+0.005, d_sc+0.005, s, fontsize=7)
        print(train_shots_dither_dice_ordered[i], f_sc, d_sc)
        
    for i,s in enumerate(test_shots_dither_dice_ordered):
        f_sc, d_sc = federico_test_scores[i], dice_test_scores[i]
        if s in mail_to_federico:
            ax.scatter(f_sc, d_sc, alpha=1.0, c='red', edgecolors='none') #, label='test'
        else:
            ax.scatter(f_sc, d_sc, alpha=.1, c='red', edgecolors='none')
        ax.text(f_sc+0.0075, d_sc-0.0075, s, fontsize=7)
        print(test_shots_dither_dice_ordered[i], f_sc, d_sc)
    plt.title('Overall scores for each shot')
    # plt.legend(loc=2)
    plt.show()

if __name__ == '__main__':
    main()