__author__ = 'cipriancorneanu'

import scipy.io
import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import preprocess.processor.encoder as enc

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

def load_fake(path, fname):
    if os.path.exists(path+fname):
        return cPickle.load(open(path+fname, 'rb'))

    files = [f for f in os.listdir(path)]
    files = [files[i] for i in  np.argsort([int(f.split('.')[0]) for f in os.listdir(path) if f.endswith('.mat')])]

    dt = []
    for file in files:
        f = scipy.io.loadmat(path+file)
        dt.append({'lms': np.asarray(f['out'][0][0], dtype=np.uint16), 'emos':f['out'][0][1]})

    dt = process_fake(dt)
    cPickle.dump(dt, open(path+fname, 'wb'),  cPickle.HIGHEST_PROTOCOL)
    return dt

def process_fake(dt):
    split =  [split_seq(person['emos']) for person in dt]
    return [ [pers['lms'][s] for s in slice] for pers,slice in zip(dt, split)]

def emo_mapping(label):
    labels = {'act_ANGRY':0, 'act_CONTEMPT':1, 'act_DISGUST':2, 'act_HAPPY':3, 'act_SAD':4, 'act_SURPRISED':5,
    'fake_ANGRY':6, 'fake_CONTEMPT':7, 'fake_DISGUST':8, 'fake_HAPPY':9, 'fake_SAD':10, 'fake_SURPRISED':11}
    return labels[label]


def split_seq(emos):
    labels = ['act_ANGRY', 'act_CONTEMPT', 'act_DISGUST', 'act_HAPPY', 'act_SAD', 'act_SURPRISED',
    'fake_ANGRY', 'fake_CONTEMPT', 'fake_DISGUST', 'fake_HAPPY', 'fake_SAD', 'fake_SURPRISED']

    # Extract emo labels
    emos = np.asarray([e[0][0] for e in emos])

    # Split according to labels
    return [emos==l for i,l in enumerate(labels)]

if __name__ == '__main__':
    path2geoms = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    path2ims = '/Users/cipriancorneanu/Research/data/fake_emotions/samples/'
    path2save = '/Users/cipriancorneanu/Research/data/fake_emotions/results/'

    # Load baseline
    load_baseline(path2geoms, 'femo_pca.pkl')


    # Load geoms
    dt = load_fake(path2geoms, 'femo.pkl')

    # Load images
    subjects = sorted([f for f in os.listdir(path2ims)])
    for subject in subjects:
        emos = [f for f in os.listdir(path2ims+subject)]

        for emo in emos:
            frames, _ = reader.get_files(path2ims+subject+'/'+emo)
            tokens = [int(x.split('.')[0][5:]) for x in frames]
            frames = [frames[x] for x  in np.argsort(tokens)]

            idx_emo = emo_mapping(emo)
            idx_pers = int(subject)-1

            for frame, lms in zip(frames[150:200:10], dt[idx_pers][idx_emo][150:200:10]):
                im = reader.read_image(path2ims+'/'+subject+'/'+emo+'/'+ frame)
                fig, ax = plt.subplots()
                ax.scatter(lms[:,0], lms[:,1])
                ax.imshow(im, cmap='Greys',  interpolation='nearest')
                #plt.text(10,10, color='w')

                #plotter.plot_qualitative(ax, lms, im, label[0][0])
                plt.savefig(path2save + subject + emo + frame)