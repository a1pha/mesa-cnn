import numpy as np
import pandas as pd
import glob
import pickle

# Directory where actigraphy files are located
actigraphy_dir = "/Users/ajadhav0517/Box/mesa/actigraphy/"
actigraphy_files = glob.glob(actigraphy_dir + '*.csv')

# Directory where outcomes files are located
outcomes1 = "/Users/ajadhav0517/Box/mesa/mesa_nhlbi/Primary/Exam5/Data/mesae5_drepos_20151101.csv"
outcomes = "/Users/ajadhav0517/Box/mesa/datasets/mesa-sleep-dataset-0.2.0.csv"
outcomes = pd.read_csv(outcomes)
outcomes = outcomes.set_index('mesaid')


def return_mesa_id(s):
    try:
        first = "/Users/ajadhav0517/Box/mesa/actigraphy/mesa-sleep-";
        last = '.csv';
        start = s.index(first) + len( first )
        end = s.index(last, start)
        return int(s[start:end])
    except ValueError:
        return 0


def get_activity(fn):
    '''Return numpy array of activity values from the file given by fn'''
    df = pd.read_csv(fn)
    return np.array(df['activity'])


def filter_activity(vec, min_hours=22, max_days=5):
    '''Return activity with invalid days removed;
    invalid days are those with activity in fewer than 22 hours.
    This is somewhat clinically motivated'''

    # nan -> zero; days with too many nans will be removed
    vec[np.isnan(vec)] = 0

    # count number of days in vector
    # note that these files have 30 second epochs, so 2880 samples per hour
    numdays = len(vec) // 2880

    # convert to minutes x hours x days
    activity_matrix = np.reshape(vec[:numdays * 2880], (120, 24, -1), order='F')

    # remove days without SOME activity in at least 22 hours
    valid_days = np.sum(np.sum(activity_matrix, axis=0) > 0, axis=0) >= min_hours

    # return flattened array with invalid days removed
    flat = activity_matrix[:, :, valid_days].flatten(order='F')[:max_days * 2880]

    if len(flat) >= 2880*5:
        return flat[:2880*5]
    else:
        return None


def create_data_dict(outcomes_desired, actigraphy_files, outcome_file):
    data_dict = {}
    for patient in actigraphy_files:
        mesaid = return_mesa_id(patient)
        if mesaid not in outcome_file.index.tolist():
            continue
        outcomes_valid = np.zeros(len(outcomes_desired))
        i = 0
        for outcome in outcomes_desired:
            if np.isnan(outcome_file.at[mesaid, outcome]):
                outcomes_valid[i] = 1
            i += 1
        if np.sum(outcomes_valid) != 0:
            continue
        elif filter_activity(get_activity(patient)) is None:
            continue
        else:
            print(patient)
            j = 0
            outcomes_values = np.zeros(len(outcomes_desired))
            for outcome in outcomes_desired:
                outcomes_values[j] = outcome_file.at[mesaid, outcome]
                j += 1
            data_dict[patient] = outcomes_values
    return data_dict

if __name__ == "__main__":
    outcome_list = ['epslpscl5c']
    data_dict = create_data_dict(outcome_list, actigraphy_files, outcomes)
    with open('data_dict.pickle', 'wb') as write:
        pickle.dump(data_dict, write, protocol=pickle.HIGHEST_PROTOCOL)

