__author__ = 'cipriancorneanu'

from toolkit import read_femo_sift, leave_one_out_femo
from sklearn.cluster import KMeans
import numpy as np
import cPickle

#TODO: 1. Load data function
#TODO: 2. Partition function (leave-on-out)
#TODO: 3. k-means grid search
#TODO: 4. Linear SVM for 6 FEs on all representations

class BoVWFramework():
    def __init__(self, path, params):
        self.path = path
        self.k_means_params = params
        self.train, self.test = leave_one_out_femo()

    def k_means(self):
        (X, y) = read_femo_sift(self.path)

        for partition, (trp, tep) in enumerate(zip(self.train[:1], self.test[:1])):
            # Select train/test data
            X_tr, y_tr = (X[trp], y[trp])
            X_te, y_te = (X(tep), y(tep))

            # Perform k-means parameter search
            for n in self.k_means_params['n_clusters']:
                print'Partition:{} n_clusters:{}'.format(partition, n)

                #Select 10% of all data
                pool_indices = np.random.randint(0,X.shape[0], (1, int(X.shape[0]/10)))
                X = X[pool_indices]

                # Perform k-means on all data
                kmeans = KMeans(n_clusters=n, random_state=0).fit(X)

                # Build representation for each partition
                X_tr = self.counter(X_tr, kmeans)
                X_te = self.counter(X_te, kmeans)

                # Dump representation
                cPickle.dump({'kmeans': kmeans, 'X_tr':X_tr, 'y_tr':y_tr, 'X_te':X_te, 'y_te':y_te},
                             open(self.path+'kmeans_' + str(partition) + '_' + str(n) + 'pkl', 'wb'),
                             cPickle.HIGHEST_PROTOCOL)

    def classification(self, data, partition, n):
        data = cPickle(open(path+'kmeans_' + str(partition) + '_' + str(n) + 'pkl', 'rb'))

        # Perform some classification

        # Evaluate
        pass

    def quantizer(self, observations, kmeans):
        n_words = kmeans.labels
        histo = np.zeros((len(observations), n_words))

        [histo[kmeans.predict(word)] for word in obs for obs in observations ]

        return histo


if __name__ == '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
    femo = cPickle.load(open(path+'femo_sift.pkl', 'rb'))

    # Concatenate data
    femo = concatenate(femo)

    # Split data
    path = 'some_path'
    bovw = BoVWFramework(path, {'n_clusters': [100, 500, 1000, 5000, 10000]})

    bovw.k_means()


