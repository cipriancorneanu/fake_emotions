__author__ = 'cipriancorneanu'

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imresize
from experiments.data import FakeEmo
import getopt
import sys

def augmenter(images, inits, cascade, n_augs):
    inits = np.tile(cascade._decode_parameters(inits), (n_augs, 1, 1))
    angles = np.random.uniform(low=-np.pi/4.0, high=np.pi/4.0, size=len(inits))
    disps = np.random.uniform(low=0.95, high=1.05, size=(len(inits), 2))
    scales = np.random.uniform(low=0.85, high=1.15, size=len(inits))
    mapping = np.tile(np.array(range(len(images)), dtype=np.int32), (n_augs,))
    for i in range(len(inits)):
        an, sc, dx, dy = angles[i], scales[i], disps[i][0], disps[i][1]
        mn = np.mean(inits[i, ...], axis=0)[None, :]
        inits[i, ...] = np.dot(
            inits[i, ...] - mn,
            sc * np.array([[np.cos(an), -np.sin(an)], [np.sin(an), np.cos(an)]], dtype=np.float32)
        ) + mn * [dx, dy]

    return cascade._encode_parameters(inits), mapping

def apply_model(model, images, steps=None):
    return model.align(images, num_steps=steps, save_all=True, augmenter=augmenter, n_augs=25)

def apply(model, o_ims):
    o_size, t_size, r_size = (224, 200, 150)
    offset = (t_size - r_size)/2

    # Target images
    t_ims = np.zeros((len(o_ims), t_size, t_size))

    # Get resize images
    t_ims[:, offset:-offset, offset:-offset] = np.asarray([imresize(im, (r_size, r_size)) for im in o_ims], dtype = np.uint8)

    # Apply model
    predictions = apply_model(model, t_ims, steps=5)
    scale = float(o_size)/r_size

    # Extract predictions for last step and compute mean
    n_inst, n_pars = len(o_ims), predictions[0][-1][0].shape[1]

    # Reshape and rescale
    preds = scale*(np.mean(predictions[0][-1][0].reshape((-1, n_inst, n_pars)), axis=0) - offset)

    return preds

def extract_geometry(path2model, path2faces, path2save, start_person, stop_person):
    femo = FakeEmo('')
    data = [[None for x in range(0,femo.n_classes)] for y in range(0,femo.n_persons)]

    model = cPickle.load(open(path2model+'continuous_300w.pkl', 'rb'))

    for p_key in range(start_person, stop_person):
        predictions = []
        for t_key in range(0,femo.n_classes):
            print 'person:{} target:{}'.format(str(p_key),t_key)

            fname = path2faces+'femo_extracted_faces_'+str(p_key)+'_'+str(t_key)+'.pkl'

            if os.path.exists(fname):
                ims =  cPickle.load(open(fname, 'rb'))
                preds = apply(path2model, ims[:2])
                data[p_key-1][t_key] = preds

                # Visualize
                for i,(p, im) in enumerate(zip(preds, ims[:2])):
                    # Show
                    fig, (ax) = plt.subplots(1, 1)
                    ax.imshow(np.squeeze(im))
                    ax.scatter(p.reshape(68,-1)[:,1], p.reshape(68,-1)[:,0])

                    plt.savefig('./prediction'+str(p_key)+'_'+str(t_key)+'.png')
            else:
                print 'Nothing to load'
    print('Dumping data to ' + path2save)

    cPickle.dump(
        ims, open(path2save+'femo_geom_csdm'+str(start_person)+'_'+str(stop_person)+'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL
    )

def run_extract_geometry(argv):
    opts, args = getopt.getopt(argv, '')
    (path2model, path2faces, path2save, start, stop) = \
        (
            args[0], args[1], args[2], int(args[3]), int(args[4])
        )

    extract_geometry(path2model, path2faces, path2save, start, stop)

if __name__ == '__main__':
    run_extract_geometry(sys.argv[1:])