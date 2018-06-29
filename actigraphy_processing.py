import numpy as np
import pandas as pd
import glob
import csv
import os

# Directory where actigraphy files are located
actigraphy_dir = "/Users/ajadhav0517/Box/mesa/actigraphy/"
actigraphy_files = glob.glob(actigraphy_dir + '*.csv')


# Some SIMPLE helper functions to take a first pass at actigraphy data


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


os.remove("Data.csv")
for x in actigraphy_files:
    entry = filter_activity(get_activity(x))
    with open("Data.csv", "a") as fp:
        if entry is not None:
            entry = np.insert(entry, 0, return_mesa_id(x), axis=0)
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(entry)
            print(x)