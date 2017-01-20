__author__ = 'cipriancorneanu'

import os
import scipy.ndimage
import scipy.io
import numpy as np
import av

def read_folder(path, sorter):

    # Get files from folder
    files, extensions = get_files(path, sorter)

    #Read according to extension
    return [read_txt(path+file) if ext=='txt' else
            read_avi(path+file) if ext=='avi' else
            read_mat(path+file) if ext=='mat' else
            read_image(path+file) for ext, file in zip(extensions, files)]

def get_files(path, sorter=None):
    # Get files from path by filtering hidden and directories
    files = [f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                    and not f.startswith('.')]

    if sorter: files = sorter(files)
    extensions = [f.split('.')[-1] for f in files]

    return (files, extensions)

def read_mat(fname):
    return scipy.io.loadmat(fname)

def read_avi(fname, mode='L'):
    container = av.open(fname)
    video = next(s for s in container.streams if s.type == b'video')

    seq = []
    for packet in container.demux(video):
        for frame in packet.decode():
            im = np.asarray(frame.to_image().convert(mode), dtype=np.uint8)
            seq.append(im)
    return seq

def read_image(fname, mode='L'):
    return np.asarray(scipy.ndimage.imread(fname, mode), dtype=np.uint8)

def read_txt(fname, start=0, stop=None):
    # Read lines from text file from start to stop
    with open(fname) as f:
        return [line.split() for line in f][start:stop]