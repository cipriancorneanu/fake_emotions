__author__ = 'cipriancorneanu'

from sklearn.cluster import KMeans
import numpy as np
import cPickle
from data import Femo

#TODO: 4. Linear SVM for 6 FEs on all representations

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
            pool_indices = np.random.randint(0,X.shape[0], (1, int(X.shape[0]/10)))
            X = np.squeeze(X[pool_indices])

            # Perform k-means on all data
            kmeans.append(KMeans(n_clusters=n, random_state=0).fit(X))

        return kmeans

    '''
    def build_bow(self, kmeans, X):
        return [for x in X]
            # Build representation for each partition
            X_tr = self.counter(X, kmeans)

        return bow

    def quantizer(self, observations, kmeans):
        n_words = kmeans.labels
        histo = np.zeros((len(observations), n_words))

        [histo[kmeans.predict(word)] for word in obs for obs in observations ]

        return histo

    def classification(self, data, partition, n):
        data = cPickle(open(path+'kmeans_' + str(partition) + '_' + str(n) + 'pkl', 'rb'))

        # Perform some classification

        # Evaluate
        pass
    '''

if __name__ == '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
    femo = cPickle.load(open(path+'femo_sift.pkl', 'rb'))

    femo_sift = Femo(path)
    bovw = BoVWFramework(path, n_clusters = [100, 500, 1000])

    # Load data
    data = femo_sift.load()

    # Leave one out
    for leave in range(0,femo_sift.n_persons):
        print 'Leave {} out'.format(leave)

        # Split data
        (X_tr, y_tr), (X_te, y_te) = femo_sift.leave_one_out(data, leave)

        # Compute kmeans and dump
        kmeans = bovw.k_means(np.concatenate(X_tr))

        print '     Dump k-means'
        cPickle.dump({'kmeans': kmeans},
                    open(path + 'kmeans_leave_' + str(leave) + '.pkl', 'wb'),
                     cPickle.HIGHEST_PROTOCOL)