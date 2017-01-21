__author__ = 'cipriancorneanu'

import scipy.io
import os
import numpy as np
import cPickle
import reader
import plotter
import matplotlib.pyplot as plt


def load_fake(path, fname):
    if os.path.exists(path+fname):
        return cPickle.load(open(path+fname, 'rb'))

    files = sorted([f for f in os.listdir(path)])

    dt = []
    for file in files:
        f = scipy.io.loadmat(path+fname)
        dt.append({'lms': np.asarray(f['out'][0][0], dtype=np.uint16), 'emos':f['out'][0][1]})

    return dt

def process_fake(dt):
    split =  [split_seq(person['emos']) for person in dt]

    #TODO Split data into classes (subject, emo, fake/true)
    pass

def split_seq(emos):
    labels = ['act_ANGRY', 'act_CONTEMPT', 'act_DISGUST', 'act_HAPPY', 'act_SAD', 'act_SURPISED',
    'fake_ANGRY', 'fake_CONTEMPT', 'fake_DISGUST', 'fake_HAPPY', 'fake_SAD', 'fake_SURPISED']

    # Extract emo labels
    emos = np.asarray([e[0][0] for e in emos])

    # Split according to labels
    return [emos==l for i,l in enumerate(labels)]

if __name__ == '__main__':
    path2geoms = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    path2ims = '/Users/cipriancorneanu/Research/data/fake_emotions/samples/'
    path2save = '/Users/cipriancorneanu/Research/data/fake_emotions/results/'

    # Load geoms
    dt = load_fake(path2geoms, 'femo.pkl')

    # Split geoms
    smth = process_fake(dt)

    # Load images
    subjects = sorted([f for f in os.listdir(path2ims)])
    for subject in subjects:
        emos = [f for f in os.listdir(path2ims+subject)]

        for emo in emos:
            frames, _ = reader.get_files(path2ims+subject+'/'+emo)
            tokens = [int(x.split('.')[0][5:]) for x in frames]
            frames = [frames[x] for x  in np.argsort(tokens)]

            for frame, lms, label in zip(frames[:20], dt[3]['lms'][:20], dt[3]['emos'][:20]):
                im = reader.read_image(path2ims+'/'+subject+'/'+emo+'/'+frame)
                fig, ax = plt.subplots()
                ax.scatter(lms[:,0], lms[:,1])
                ax.imshow(im, cmap='Greys',  interpolation='nearest')
                plt.text(10,10,str(label[0][0]), color='w')

                #plotter.plot_qualitative(ax, lms, im, label[0][0])
                plt.savefig(path2save + label[0][0] + frame)
