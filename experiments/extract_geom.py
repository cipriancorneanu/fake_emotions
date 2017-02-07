__author__ = 'cipriancorneanu'

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
from cascaded.cascade import csdm

def augmenter(images, inits, cascade, n_augs):
    inits = np.tile(cascade._decode_parameters(inits), (n_augs, 1, 1))
    angles = np.random.uniform(low=-np.pi/4.0, high=np.pi/4.0, size=len(inits))
    disps = np.random.uniform(low=0.95, high=1.05, size=(len(inits), 2))
    scales = np.random.uniform(low=0.9, high=1.1, size=len(inits))
    mapping = np.tile(np.array(range(len(images)), dtype=np.int32), (n_augs,))
    for i in range(len(inits)):
        an, sc, dx, dy = angles[i], scales[i], disps[i][0], disps[i][1]
        mn = np.mean(inits[i, ...], axis=0)[None, :]
        inits[i, ...] = np.dot(
            inits[i, ...] - mn,
            sc * np.array([[np.cos(an), -np.sin(an)], [np.sin(an), np.cos(an)]], dtype=np.float32)
        ) + mn * [dx, dy]

    return cascade._encode_parameters(inits), mapping

def apply(mpath, images, steps=None):
    model = cPickle.load(open(mpath, 'rb'))
    predictions = model.align(images, num_steps=steps, save_all=True)
    return predictions

if __name__ == '__main__':
    # Load results and show
    model_file, results_file = ('continuous_300w.pkl', 'continuous_300w_results.pkl')
    path2faces = '/Users/cipriancorneanu/Research/data/fake_emotions/extracted_faces/'
    path2model = '/Users/cipriancorneanu/Research/data/300w/'

    for f in [f for f in os.listdir(path2faces) if not f.startswith('.')]:

        # Load image
        ims = cPickle.load(open(path2faces+f, 'rb'))

        # Apply model
        prediction = apply(path2model+model_file, ims, steps=3)

        pass
