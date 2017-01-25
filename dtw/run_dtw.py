__author__ = 'cipriancorneanu'

import os
from preprocess.reader.loader import load_baseline
from fastdtw import fastdtw
from sklearn.model_selection import train_test_split
import cPickle
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.metrics import accuracy_score

def dtw(ts_seq):
    n = len(ts_seq)
    dists = np.zeros((len(ts_seq), len(ts_seq)))
    iarr, jarr = np.triu_indices(n=len(ts_seq), m=len(ts_seq))

    for i,j in zip(iarr, jarr):
        print '({}, {})'.format(i,j)
        if ts_seq[i].size and ts_seq[j].size and i!=j:
            d, _ = fastdtw(ts_seq[i], ts_seq[j], dist=euclidean)
            dists[i,j] = d
            dists[j,i] = d

    cPickle.dump(dists, open('/Users/cipriancorneanu/Research/data/fake_emotions/geoms/dtw_results.pkl', 'wb'),
                 cPickle.HIGHEST_PROTOCOL
                 )

def leave_one_out(n_pers, n_seq):
    test = [ range(x*6, x*6+6)+range(x*6+324, x*6+330) for x in np.arange(0,n_pers)]
    train = [ list(set(range(0,n_pers*n_seq))-set(t)) for t in test]

    return (train, test)

if __name__ == '__main__':
    # Build partitions
    #train_part, test_part = leave_one_out(54, 12)

    # Load data
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    (X, y) = load_baseline(path, 'femo_baseline.pkl')

    dtw(X)

    '''
    for trp, tep in zip(train_part, test_part):

        # Select distances

        # Perform k-NN classification

        # Compute distances
        if os.path.exists(path+'dtw_results_short.pkl'):
            dtw_dists = cPickle.load(open(path+'dtw_results.pkl', 'rb'))
        else:
            dtw_dists = dtw(X_train, X_test)

        # Perform 1-NN classification
        y_test_pred = np.zeros((1,len(dtw_dists)))
        for i,d in enumerate(dtw_dists):
            dists = [x[0]for x in d]
            idx = dists.index(min(dists))
            y_test_pred[i] = y_train[i]


        # Evaluate
        #print accuracy_score(y_test, y_test_pred)
    '''



