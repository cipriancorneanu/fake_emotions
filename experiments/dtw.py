__author__ = 'cipriancorneanu'

import os
from preprocess.reader import loader
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

if __name__ == '__main__':
    # Build partitions
    #train_part, test_part = leave_one_out(54, 12)

    # Load processed geometries
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'

    #Load processed geometries
    data = cPickle.load(open(path+'femo_geom_raw.pkl', 'rb'))

    # Process
    y = np.asarray([sample[0] for sample in y])

    # Dump
    cPickle.dump((X,y), open(path+'femo_geom_proc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)




