__author__ = 'cipriancorneanu'

from bow import *
from classify import *
from data import *
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.stats import norm
from sift import DescriptorSift

def normal_pool(N, n):
    # Create normal distribution with 95% in the central 50% of the seq
    prob = norm.pdf(range(0,N), loc=N/2, scale=N/8)
    prob = prob/np.sum(prob)

    # Return sorted pooling
    return np.sort(np.random.choice(range(0,N), size = n, replace = False, p = prob))

def extract_descriptor(sift, im, lm):
    return sift.extract(im, lm, np.arange(0, im.shape[0]), {'num_bins':8, 'window_sizes':32})

def load_sift(path, fname):
    pass

def order_geometries(geoms):
    indices, nindices = (range(0,12), [3,4,8,5,2,0,9,10,7,11,1,6])

    for i_p, person in enumerate(geoms):
        new_person = [None]*12
        for i, ni in zip(indices, nindices):
            new_person[i] = person[ni]
        geoms[i_p] = new_person

    return geoms

def read_sift(path2faces, path2geoms, persons):
    if os.path.exists(path2geoms+'femo_geom_raw.pkl'):
        geoms =  cPickle.load(open(path2geoms+'femo_geom_raw.pkl', 'rb'))

    n_classes = 12

    data = [[None for _ in range(n_classes)] for _ in range(0, len(persons))]
    sift = DescriptorSift()
    for i_p, p_key in enumerate(persons):
        for t_key in range(0,2):#n_classes):
            print 'person:{} target:{}'.format(str(p_key),t_key)

            fname = path2faces+'femo_extracted_faces_'+str(p_key)+'_'+str(t_key)+'.pkl'
            if os.path.exists(fname):
                ims =  cPickle.load(open(fname, 'rb'))
                data[i_p][t_key] = extract_descriptor(sift, ims, geoms[p_key-1][t_key])
    return data

def full_bow(path2data, fname):
    n_persons = 2
    n_clusters = [50, 100, 200]
    vars = [0.9, 0.95, 0.99]

    # Load data
    femo = FakeEmo(path2data)
    data = femo.load(fname)[:n_persons]

    clf = LinearSVC()
    results_frame = np.zeros((n_persons, len(n_clusters), len(vars)))
    results_seq = np.zeros((n_persons, len(n_clusters), len(vars)))

    for leave_out in range(0,n_persons):
        print 'Leave {}'.format(leave_out)
        # Prepare sequences
        (X_ftr, y_ftr), (X_fte, y_fte) = femo.leave_one_out(data, leave_out, format='frames')
        (X_str, y_str), (X_ste, y_ste) = femo.leave_one_out(data, leave_out, format='sift_sequences')

        for i_v, var in enumerate(vars):
            print '     Variance {}'.format(var)
            # Perform PCA
            pca = PCA(n_components=var, whiten=True).fit(np.concatenate(X_ftr)[::5])
            X_ftr_pca = [pca.transform(x) for x in X_ftr]
            X_fte_pca = [pca.transform(x) for x in X_fte]

            X_str_pca = [pca.transform(x) for x in X_str]
            X_ste_pca = [pca.transform(x) for x in X_ste]

            # k-means
            (_, feat_X_ftr, feat_X_fte) = generate_all(X_ftr_pca, X_fte_pca, n_clusters)
            (_, feat_X_str, feat_X_ste) = generate_all(X_str_pca, X_ste_pca, n_clusters)

            # Per frame representation
            print '             Classification with frame features'
            for i_n, (x_tr, x_te) in enumerate(zip(feat_X_ftr, feat_X_fte)):
                # Train with 20% of the data
                clf.fit(x_tr['feats'][::5], y_ftr[::5])

                # Predict
                y_te_pred = clf.predict(x_te['feats'])

                # Eval
                results_frame[leave_out][i_n][i_v] = accuracy_score(y_fte, y_te_pred)

            # Per sequence representation
            print '             Classificaton with sequence features'
            for i_n, (x_tr, x_te) in enumerate(zip(feat_X_str, feat_X_ste)):
                # Train
                clf.fit(np.asarray(x_tr['feats']), y_str)

                # Predict
                y_te_pred = clf.predict(np.asarray(x_te['feats']))

                # Eval
                results_seq[leave_out][i_n][i_v] = accuracy_score(y_ste, y_te_pred)

    print np.mean(results_frame, axis = 0)
    print np.mean(results_seq, axis = 0)


if __name__ == '__main__':
    path2faces = '/home/corneanu/data/fake_emotions/extracted_faces/'
    path2geoms = '/home/corneanu/data/fake_emotions/geoms/'
    dt = read_sift(path2faces, path2geoms, range(6,20))

    cPickle.dump(open('home/corneanu/data/fake_emotions/sift/'+'femo_sift_sem_6_20'))
