__author__ = 'cipriancorneanu'

from reader.loader import load_fake
from processor.partitioner import *
import processor.encoder as enc
from cascaded.toolkit import procrustes, linalg
import numpy as np


if __name__ == '__main__':
    # Load data
    path2data = '/Users/cipriancorneanu/Research/data/fake_emotions/geoms/'
    data = np.asarray(load_fake(path2data, 'femo.pkl'))

    # Split in the two classes
    true = np.concatenate([x[:6] for x in data])
    fake = np.concatenate([x[6:] for x in data])

    # Prepare for processing
    uncoded = list(true) + list(fake)

    # Concatenate sequences
    uncoded, uncoded_slices = (np.concatenate(uncoded), slice(uncoded))

    # Encode (procustes + pca)
    coded, mean, T = enc.encode_parametric(uncoded)

    pass