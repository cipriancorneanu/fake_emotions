__author__ = 'cipriancorneanu'

from reader.loader import load_fake
from processor.partitioner import slice
import processor.encoder as enc
import os
import numpy as np
import cPickle
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_baseline(path, fname):
    if os.path.exists(path+fname):
        return cPickle.load(open(path+fname, 'rb'))

    # Load data
    path2data = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    data = np.asarray(load_fake(path, 'femo.pkl'))

    # Split in the two classes
    true = np.concatenate([x[:6] for x in data])
    fake = np.concatenate([x[6:] for x in data])

    # Prepare for processing
    uncoded = list(true) + list(fake)

    # Concatenate sequences
    uncoded, uncoded_slices = (np.concatenate(uncoded), slice(uncoded))

    # Encode (procustes + pca)
    coded, mean, T = enc.encode_parametric(uncoded)

    # Keep non-rigid params only
    coded = coded[:,:4]

    # Split back
    data = [coded[us] for us in uncoded_slices]
    target = np.asarray(
        np.concatenate([np.ones((len(uncoded_slices)/2,1)), 0*np.ones((len(uncoded_slices)/2,1))]),
        dtype=np.int
    )

    cPickle.dump((data, target), open(path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)

    return (data, target)

def split_data(data, target, split=0.2):
    return train_test_split(data, target, test_size=split, random_state=0)

if __name__ == '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    (data, target) = load_baseline(path, 'femo_baseline.pkl')

    # Split into train test
    X_train, X_test, y_train, y_test = split_data(data, target)

    # Classify
    res = []
    for i,te in enumerate(X_test):
        print 'Classifying test sample ' + str(i)
        res.append([fastdtw(te, tr, dist=euclidean) for tr in X_train])

    cPickle.dump(res, open(path+'results.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)



