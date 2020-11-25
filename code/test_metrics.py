import pandas as pd
from helper_funcs import *
import sys

def test(path):
    exit_code = 0
    # ------------------ TEST 1 ---------------------
    fshot_labeled = pd.read_csv(path+'/TCV_00001_labeled.csv')
    fshot_det = pd.read_csv(path+'/TCV_00001_LSTM_det.csv')
    #These two files should be exactly equal, apart from the fact that the column names with ELMs and LHD states are different.

    dice_cf = np.round(dice_coefficient(fshot_det['LHD_det'].values, fshot_labeled['LHD_label'].values), 3)
    conf_mat = elm_conf_matrix(fshot_det['elm_det'].values, fshot_labeled['elm_label'].values)
    conf_met = np.round(conf_metrics(*conf_mat), 3)
    dice_correct = np.ones(4)
    conf_mat_correct = np.asarray([4, 0, 3, 0])
    conf_met_correct = np.asarray([1, 1, 1, 1, 0, 0, 0])
    
    try:
        assert(np.array_equal(dice_cf, dice_correct))
        assert(np.array_equal(conf_mat, conf_mat_correct))
        assert(np.array_equal(conf_met, conf_met_correct))
    except:
        print('Failure on test 1.')
        print('Dice coef. should have been', dice_correct, 'was', dice_cf)
        print('Conf. matrix should have been', conf_mat_correct, 'was', conf_mat)
        print('Normalized conf. matrix should have been', conf_met_correct, 'was', conf_met)
        exit_code = 1
        
    # ------------------ TEST 2 ---------------------
    fshot_labeled = pd.read_csv(path+'/TCV_00002_labeled.csv')
    fshot_det = pd.read_csv(path+'/TCV_00002_LSTM_det.csv')
    #These two files are not equal!
    
    dice_cf = np.round(dice_coefficient(fshot_det['LHD_det'].values, fshot_labeled['LHD_label'].values), 3)
    conf_mat = elm_conf_matrix(fshot_det['elm_det'].values, fshot_labeled['elm_label'].values)
    conf_met = np.round(conf_metrics(*conf_mat), 3)
    dice_correct = np.asarray([0, .711, 1, .608])
    conf_mat_correct = np.asarray([3, 2, 8, 0])
    conf_met_correct = np.asarray([1., 0.8, 0.6, 1., 0.2, 0., 0.4])
    
    try:
        assert(np.array_equal(dice_cf, dice_correct))
        assert(np.array_equal(conf_mat, conf_mat_correct))
        assert(np.array_equal(conf_met, conf_met_correct))
    except:
        print('Failure on test 2.')
        print('Dice coef. should have been', dice_correct, 'was', dice_cf)
        print('Conf. matrix should have been', conf_mat_correct, 'was', conf_mat)
        print('Normalized conf. matrix should have been', conf_met_correct, 'was', conf_met)
        exit_code = 1
    
    # ------------------ TEST 3 ---------------------
    #Test with data from a single file, ie, exactly same input arguments for detection and label
    dice_cf = np.round(dice_coefficient(fshot_labeled['LHD_label'].values, fshot_labeled['LHD_label'].values), 3)
    conf_mat = elm_conf_matrix(fshot_labeled['elm_label'].values, fshot_labeled['elm_label'].values)
    conf_met = np.round(conf_metrics(*conf_mat), 3)
    dice_correct = np.ones(4)
    conf_mat_correct = np.asarray([3, 0, 9, 0])
    conf_met_correct = np.asarray([1, 1, 1, 1, 0, 0, 0])
    
    try:
        assert(np.array_equal(dice_cf, dice_correct))
        assert(np.array_equal(conf_mat, conf_mat_correct))
        assert(np.array_equal(conf_met, conf_met_correct))
    except:
        print('Failure on test 3')
        print('Dice coef. should have been', dice_correct, 'was', dice_cf)
        print('Conf. matrix should have been', conf_mat_correct, 'was', conf_mat)
        print('Normalized conf. matrix should have been', conf_met_correct, 'was', conf_met)
        exit_code = 1
        
    if exit_code == 0: 
        print('Tests executed successfuly.')
    else:
        print('Some test(s) failed.')
    return exit_code
    
def main():
    exit(test(sys.argv[1]))

if __name__ == "__main__":
    # execute only if run as a script
    main()
