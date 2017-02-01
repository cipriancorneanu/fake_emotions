__author__ = 'cipriancorneanu'

from sklearn.cluster import KMeans
import numpy as np
import cPickle
from data import Femo

class BoVWFramework():
    def __init__(self, path, n_clusters):
        self.path = path
        self.n_clusters = n_clusters

    def k_means(self, X):
        kmeans = []
        # Perform k-means parameter search
        for i,n in enumerate(self.n_clusters):
            print'  Computing kmeans with {} clusters'.format(n)

            #Select 10% of all data
            pool_indices = np.random.randint(0,X.shape[0], (1, int(X.shape[0]/20)))
            X = np.squeeze(X[pool_indices])

            # Perform k-means
            kmeans.append(KMeans(n_clusters=n, random_state=0).fit(X))

        return kmeans

    def build_bow(self, kmeans, X):
        histo = np.zeros((len(X), kmeans.n_clusters), dtype = np.int16)

        for i, (frame,h) in enumerate(zip(X, histo)):
            words = kmeans.predict(frame)
            for w in words:
                h[w] += 1

        return histo

if __name__ == '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'

    femo_sift = Femo(path)
    bovw = BoVWFramework(path, n_clusters = [50,100,200])

    # Load data
    data = femo_sift.load()

    # Leave one out
    for leave in range(0, femo_sift.n_persons):
        print 'Leave {} out'.format(leave)

        # Split data
        (X_tr, y_tr), (X_te, y_te) = femo_sift.leave_one_out(data, leave)

        # Compute kmeans and dump
        kmeans = bovw.k_means(np.concatenate(X_tr))

        print '     Dump k-means'
        cPickle.dump({'kmeans': kmeans},
                    open(path + 'kmeans_leave_' + str(leave) + '.pkl', 'wb'),
                     cPickle.HIGHEST_PROTOCOL)

        print '     Compute representation'

        # Compute representation
        for km in kmeans:
            X_tr = bovw.build_bow(km, X_tr)
            X_te = bovw.build_bow(km, X_te)

            print '     Dump representation'
            cPickle.dump({'kmeans': km, 'X_tr': X_tr, 'y_tr': y_tr, 'X_te':X_te, 'y_te':y_te},
                        open(path + str(leave) + '_' + str(km.n_clusters) + '.pkl', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)