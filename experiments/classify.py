__author__ = 'cipriancorneanu'

from data import Femo
import cPickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

def slice(labels):
    borders = [i for i,(x, x_) in enumerate(zip(labels[:-1], labels[1:])) if x!=x_]
    return [range(bmin, bmax) for bmin, bmax in zip([0]+borders[:-1], borders)]

def middle_partition(slices):
    partitions = [0.2, 0.3, 0.5]
    middle_partitions = [None]*len(partitions)

    for i,part in enumerate(partitions):
         middle_partitions[i] = [s[int(len(s)*part/2):int(len(s)*(1-part/2))] for s in slices]

    return middle_partitions

if __name__ == '__main__':

    # Init classes
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/'
    femo = Femo(path)
    clf = LinearSVC()

    # Leave-one-out
    for n_clusters in [100,500,1000]:
        for leave in range(0,femo.n_persons):
            dt = cPickle.load(open(path+str(leave)+'_'+str(n_clusters)+'.pkl', 'rb'))
            (X_tr, X_te, y_tr, y_te)= (dt['X_tr'], dt['X_te'],dt['y_tr'], dt['y_te'])

            slices = middle_partition(slice(y_tr))

            for s in slices:
                clf.fit(X_tr[::10], y_tr[::10])

                y_te_pred = clf.predict(X_te)

                print accuracy_score(y_te, y_te_pred)