__author__ = 'cipriancorneanu'

import cPickle
import os
import scipy.io
import numpy as np

def read_femo_sift(path):
    n_persons, n_classes = (54, 12)
    data = [[None for _ in range(n_classes)] for _ in range(n_persons)]

    person_keys = ['_'+str(x)+'_' for x in np.arange(1,n_persons)]
    target_keys = ['act_HAPPY', 'act_SAD', 'act_CONTEMPT', 'act_SUPRISED', 'act_DISGUST', 'act_ANGRY',
                   'fake_HAPPY', 'fake_SAD', 'fake_CONTEMPT', 'fake_SUPRISED', 'fake_DISGUST', 'fake_ANGRY']

    mapping = {'act':{'HAPPY':0, 'SAD':1, 'CONTEMPT':2, 'SUPRISED':3, 'DISGUST':4, 'ANGRY':5},
              'fake':{'HAPPY':6, 'SAD':7, 'CONTEMPT':8, 'SUPRISED':9, 'DISGUST':10, 'ANGRY':11}}

    files = [f for f in os.listdir(path)]

    # Slice by sequence
    for p_key in person_keys:
        person_seq = [f for f in files if p_key in f]

        #Slice by target
        for t_key in target_keys:
            print 'person:{} target:{}'.format(p_key[1:3],t_key)

            target_seq = [f for f in person_seq if t_key in f]

            # Extract frame numbers and compute last and use as length
            last_frame = max([ int(f.split('.')[0].split('_')[0][3:]) for f in target_seq])
            seq = [None]*last_frame

            # Read sequence
            for i,f in enumerate(target_seq):
                # Parse fname
                tokens = f.split('.')[0].split('_')
                category, fe, person, frame = (tokens[2], tokens[3], int(tokens[1]), int(tokens[0][3:])-1)
                target = mapping[category][fe]

                # Load data
                fdata = cPickle.load(open(path+f, 'rb'))
                seq[frame] = np.asarray(fdata, dtype=np.int16)

            data[person][target] = seq

    cPickle.dump(data, open(path+'femo_sift.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    return data

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
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'

    read_femo_sift(path)


