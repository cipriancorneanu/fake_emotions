__author__ = 'cipriancorneanu'

from sklearn.cluster import KMeans
import numpy as np
import cPickle
from data import FakeEmo
import getopt, sys

def generate_kmeans(X, n):
    # Pool 10% of all data
    pool_indices = np.random.randint(0,X.shape[0], (1, int(X.shape[0]/10)))
    X = np.squeeze(X[pool_indices])

    return KMeans(n_clusters=n, random_state=0).fit(X)

def grid_generate_kmeans(X, n_clusters):
    # Generate words for all parameters
    kmeans = [None]*len(n_clusters)
    for i,n in enumerate(n_clusters):
        print'          {} clusters'.format(n)
        kmeans[i] = {'n_clusters': n, 'kmeans': generate_kmeans(np.concatenate(X), n)}

    return kmeans

def generate_features(X, kmeans):
    features = np.zeros((len(X), kmeans.n_clusters), dtype = np.int16)

    for i, (frame,feat) in enumerate(zip(X, features)):
        words = kmeans.predict(frame)
        for w in words:
            feat[w] += 1

    return features

def grid_generate_features(X, kmeans):
    features = [None]*len(kmeans)
    for i,km in enumerate(kmeans):
        features[i] = {'n_clusters': km['n_clusters'], 'feats': generate_features(X, km['kmeans'])}

    return features

def generate_all(X_tr, X_te, n_clusters=[50,100,200], dump=True):
    print '     Compute kmeans'
    kmeans = grid_generate_kmeans(X_tr, n_clusters)

    print '     Compute representation'
    feat_X_tr = grid_generate_features(X_tr, kmeans)
    feat_X_te = grid_generate_features(X_te, kmeans)

    return (kmeans, feat_X_tr, feat_X_te)

def bow(path2data, path2save, n_clusters, start_person, stop_person):
    fake_emo_sift = FakeEmo(path2data)

    # Load data
    data = fake_emo_sift.load('femo_sift.pkl')[:2]

    # Leave one out
    for leave in range(start_person, stop_person):
        print 'Leave {} out'.format(leave)

        (X_tr, y_tr), (X_te, y_te) = fake_emo_sift.leave_one_out(data, leave, format='frames')
        kmeans, feat_X_tr, feat_X_te = generate_all(X_tr, X_te, n_clusters)

        print '     Dump representation'
        for km, tr, te in zip(kmeans, feat_X_tr, feat_X_te):
            cPickle.dump({'kmeans': km['kmeans'], 'X_tr': tr, 'y_tr': y_tr, 'X_te': te, 'y_te': y_te},
                        open(path2save + str(leave) + '_' + str(km['n_clusters'])+ '.pkl', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

def run_bow(argv):
    opts, args = getopt.getopt(argv, '')
    (path2data, path2save, n_clusters, start, stop) = \
        (
            args[0], args[1], [int(x) for x in args[2].split(',')], int(args[3]), int(args[4])
        )

    bow(path2data, path2save, n_clusters, start, stop)

if __name__ == "__main__":
    run_bow(sys.argv[1:])
    '''bow('/Users/cipriancorneanu/Research/data/fake_emotions/sift/',
        '/Users/cipriancorneanu/Research/data/fake_emotions/sift/',
        [10], 0, 1)'''