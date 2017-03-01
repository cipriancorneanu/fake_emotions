__author__ = 'cipriancorneanu'

import numpy as np
import os
import cPickle
from  scipy.ndimage import imread
from  scipy.misc import imresize

class FakeEmo:
    def __init__(self, path):
        self.n_persons = 54
        self.n_classes = 12
        self.path = path

        self.sift_person_keys = ['_'+str(x+1)+'_' for x in np.arange(0,self.n_persons)]
        self.target_keys = ['act_HAPPY', 'act_SAD', 'act_CONTEMPT', 'act_SURPRISED', 'act_DISGUST', 'act_ANGRY',
                       'fake_HAPPY', 'fake_SAD', 'fake_CONTEMPT', 'fake_SURPRISED', 'fake_DISGUST', 'fake_ANGRY']

    def _map_class(self, category, emo):
        map= {'act':{'HAPPY':0, 'SAD':1, 'CONTEMPT':2, 'SURPRISED':3, 'DISGUST':4, 'ANGRY':5},
              'fake':{'HAPPY':6, 'SAD':7, 'CONTEMPT':8, 'SURPRISED':9, 'DISGUST':10, 'ANGRY':11}}

        return map[category][emo]

    def load(self, fname):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))
        else:
            print 'Nothing to load'

    def read(self, path2load, path2save, fname):
        data = [[None for _ in range(self.n_classes)] for _ in range(self.n_persons)]

        files = [f for f in os.listdir(path2load)]

        # Slice by sequence
        for p_key in self.sift_person_keys:
            person_seq = [f for f in files if p_key in f]

            # Slice by target
            for t_key in self.target_keys:
                print 'person:{} target:{}'.format(p_key[1:3],t_key)

                target_seq = [f for f in person_seq if t_key in f]

                if target_seq:
                    # Extract frame numbers and compute last and use as length
                    last_frame = max([ int(f.split('.')[0].split('_')[0]) for f in target_seq])
                    seq = [None]*last_frame

                    # Read sequence
                    for i,f in enumerate(target_seq):
                        # Parse fname
                        tokens = f.split('.')[0].split('_')
                        category, fe, person, frame = (tokens[2], tokens[3], int(tokens[1])-1, int(tokens[0])-1)
                        target = self._map_class(category,fe)

                        # Load data
                        fdata = cPickle.load(open(path2load+f, 'rb'))
                        seq[frame] = np.asarray(fdata, dtype=np.int16)

                    data[person][target] = seq
                else:
                    print 'Target sequence empty'

        print('Dumping data to ' + path2save)
        cPickle.dump(data, open(path2save+fname+'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

        '''
        for i, part in enumerate([range(0,9), range(9, 18), range(18, 27), range(27,36), range(36, 45), range(45, 54)]):
            cPickle.dump([data[x] for x in part], open(path2save+fname+str(i)+'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
        '''
        return data

    def read_extracted_faces(self, path2load, path2save):
        data = [[None for _ in range(self.n_classes)] for _ in range(self.n_persons)]

        for p_key in range(1,self.n_persons+1):
            for t_key in  [f for f in os.listdir(path2load+str(p_key)+'/') if not f.startswith('.')]:
                print 'person:{} target:{}'.format(str(p_key),t_key)

                files = [f for f in os.listdir(path2load+str(p_key)+'/'+t_key+'/') if f.endswith('.png')]
                files.sort()

                ims = np.zeros((len(files),224,224), dtype=np.uint8)
                for i,f in enumerate(files):
                    im = imread(path2load+str(p_key)+'/'+t_key+'/'+f,'L')
                    ims[i,:,:] = np.asarray(imresize(im,(224, 224)), dtype=np.uint8)

                print('Dumping data to ' + path2save)
                cat, emo = t_key.split('_')
                cPickle.dump(
                    ims, open(path2save+'femo_extracted_faces_'+str(p_key)+'_'+str(self._map_class(cat, emo))+'.pkl', 'wb'),
                    cPickle.HIGHEST_PROTOCOL
                )

    def format_feat_fnames(self, path):
        fnames = [f for f in os.listdir(path) if f.endswith('.pkl')]
        roots, persons, cats, emos = zip(*[f.split('.')[0].split('_') for f in fnames])
        roots = [r.split('.')[0][5:] if r.startswith('frame') else (r.split('[')[0] if '[' in r else r ) for r in roots]
        frames = ['0'*(5 - len(r))+r for r in roots ]

        # Change names
        nfnames = [ fnm+'_'+prs+'_'+cat+'_'+emo+'.pkl' for (fnm, prs, emo, cat) in zip(frames, persons, emos, cats)]

        fnames  = [f.replace(" ", "\ ") for f in os.listdir(path) if f.endswith('.pkl')]
        for fn, nfn in zip(fnames, nfnames):
            os.system('mv '+path+fn+' '+path+nfn)

    def leave_one_out(self, data, n, format='frames'):
        all = range(0,len(data))
        train = list(set(all)-set([n]))
        test = [n]

        if format=='frames':
            return (self.prepare_frames([data[x] for x in train]), self.prepare_frames([data[x] for x in test]))
        elif format=='sift_sequences':
            return (self.prepare_sequences([data[x] for x in train]), self.prepare_sequences([data[x] for x in test], format))
        elif format=='vgg_sequences':
            return (self.prepare_sequences([data[x] for x in train], format), self.prepare_sequences([data[x] for x in test], format))

    def prepare_frames(self, data):
        X, y = ([],[])
        for i_pers,pers in enumerate([x for x in data if x is not None]):
                for i_emo,emo in enumerate([x for x in pers if x is not None]):
                        for frame in [x for x in emo if x is not None]:
                                y.append(i_emo)
                                X.append(frame)
        return (X,y)

    def prepare_sequences(self, data, format='sift_sequences'):
        X, y = ([],[])
        for i_pers,pers in enumerate([x for x in data if x is not None]):
                for i_emo,emo in enumerate([x for x in pers]):
                    if emo is not None:
                        emo = [x for x in emo if x is not None]
                        y.append(i_emo)
                        if format=='sift_sequences':
                            X.append(np.concatenate(emo))
                        elif format=='sift_sequences_per_frame':
                            X.append(np.reshape(np.asarray(emo), (len(emo), -1)))
                        elif format=='sift_sequences_avg':
                            X.append(np.mean(emo, axis=0))
                        elif format=='vgg_sequences':
                            X.append(np.asarray(emo))
                        elif format=='vgg_pool5_sequences':
                            X.append(np.asarray([np.reshape(d, -2) for d in data[0]]))
        return (X,y)
    
if __name__ == '__main__':
    path2load = '/Users/cipriancorneanu/Research/data/fake_emotions/vgg/VGGOrigFC7Fake/'
    path2save = '/home/corneanu/data/fake_emotions/extracted_faces/'

    femo = FakeEmo(path2load)
    femo.read(path2load, path2save, 'femo_vgg_fc7_')