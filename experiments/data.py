__author__ = 'cipriancorneanu'

import numpy as np
import os
import cPickle

class Femo:
    def __init__(self, path):
        self.n_persons = 54
        self.n_classes = 12
        self.path = path

        self.person_keys = ['_'+str(x+1)+'_' for x in np.arange(0,self.n_persons)]
        self.target_keys = ['act_HAPPY', 'act_SAD', 'act_CONTEMPT', 'act_SURPRISED', 'act_DISGUST', 'act_ANGRY',
                       'fake_HAPPY', 'fake_SAD', 'fake_CONTEMPT', 'fake_SURPRISED', 'fake_DISGUST', 'fake_ANGRY']

    def map_class(self, category, emo):
        map= {'act':{'HAPPY':0, 'SAD':1, 'CONTEMPT':2, 'SURPRISED':3, 'DISGUST':4, 'ANGRY':5},
              'fake':{'HAPPY':6, 'SAD':7, 'CONTEMPT':8, 'SURPRISED':9, 'DISGUST':10, 'ANGRY':11}}

        return map[category][emo]

    def load(self):
        if os.path.exists(self.path+'femo_sift.pkl'):
            return cPickle.load(open(self.path+'femo_sift.pkl', 'rb'))
        else:
            print 'Nothing to load'

    def read(self, path2save):
        data = [[None for _ in range(self.n_classes)] for _ in range(self.n_persons)]

        files = [f for f in os.listdir(path2load)]

        # Slice by sequence
        for p_key in self.person_keys:
            person_seq = [f for f in files if p_key in f]

            # Slice by target
            for t_key in self.target_keys:
                print 'person:{} target:{}'.format(p_key[1:3],t_key)

                target_seq = [f for f in person_seq if t_key in f]

                if target_seq:
                    # Extract frame numbers and compute last and use as length
                    last_frame = max([ int(f.split('.')[0].split('_')[0][3:]) for f in target_seq])
                    seq = [None]*last_frame

                    # Read sequence
                    for i,f in enumerate(target_seq):
                        # Parse fname
                        tokens = f.split('.')[0].split('_')
                        category, fe, person, frame = (tokens[2], tokens[3], int(tokens[1])-1, int(tokens[0][3:])-1)
                        target = self.map_class(category,fe)

                        # Load data
                        fdata = cPickle.load(open(path2load+f, 'rb'))
                        seq[frame] = np.asarray(fdata, dtype=np.int16)

                    data[person][target] = seq
                else:
                    print 'Target sequence empty'

        print('Dumping data to ' + path2save)

        for i, part in enumerate([range(0,9), range(9, 18), range(18, 27), range(27,36), range(36, 45), range(45, 54)]):
            cPickle.dump([data[x] for x in part], open(path2save+'femo_sift_'+str(i)+'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

        return data

    def leave_one_out(self, data, n):
        all = range(0,self.n_persons)
        train = list(set(all)-set([n]))
        test = n

        return (self.prepare([data[x] for x in train]), self.prepare([data[n]]))

    def prepare(self, data):
        X, y = ([],[])
        for i_pers,pers in enumerate([x for x in data if x is not None]):
                for i_emo,emo in enumerate([x for x in pers if x is not None]):
                        for frame in [x for x in emo if x is not None]:
                                y.append(i_emo)
                                X.append(frame)
        return (X,y)

if __name__ == '__main__':
    path2load = '/data/hupba2/Derived/FaceSIFTs/'
    path2save = '/home/corneanu/data/fake_emotions/'

    femo = Femo(path2load)
    femo.load(path2save)

