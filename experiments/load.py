__author__ = 'cipriancorneanu'

import cPickle
import os
import scipy.io

def read_femo_sift(path):
    femo_sift = None
    return femo_sift

def pkl2mat(path, fname):
    (X,y) = cPickle.load(open(path+fname+'.pkl', 'rb'))
    scipy.io.savemat(path+fname+'.mat', mdict={'X': X, 'y': y})

def mat2pkl(path, fname):
    scipy.io.loadmat()

    pass

if __name__ == '__main__':
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'

    pkl2mat(path, 'femo_geom_proc')
