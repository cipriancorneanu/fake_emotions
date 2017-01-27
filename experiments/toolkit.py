__author__ = 'cipriancorneanu'

import cPickle
import os
import scipy.io
import numpy as np

def read_femo_sift(path):
    femo_sift = None
    return femo_sift

def pkl2mat(path, fname):
    (X,y) = cPickle.load(open(path+fname+'.pkl', 'rb'))
    scipy.io.savemat(path+fname+'.mat', mdict={'X': X, 'y': y})

def mat2pkl(path, fname):
    scipy.io.loadmat()

    pass

def leave_one_out_femo():
    N = 648
    n_pers = 54
    test = [ range(x*6, (x+1)*6)+range(x*6+N/2, (x+1)*6+N/2) for x in np.arange(0,n_pers)]
    train = [ list(set(range(0,N))-set(t)) for t in test]

    return (train, test)

if __name__ == '__main__':
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'

    pkl2mat(path, 'femo_geom_proc')
