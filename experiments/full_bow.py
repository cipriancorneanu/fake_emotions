__author__ = 'cipriancorneanu'

from bow import *
from classify import *
from data import *
import sklearn.model_selection
from  sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def full_bow(path2data, fname):
    n_persons = 2
    n_clusters = [20, 50, 100]
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
            pca = PCA(n_components=var, whiten=True).fit(np.concatenate(X_ftr))
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
                # Change classes
                #(x_tr, y_tr) = change_classes(x_tr['feats'], y_ftr, mode='6classes_true')
                #(x_te, y_te) = change_classes(x_te['feats'], y_fte, mode='6classes_true')

                # Train
                clf.fit(x_tr['feats'], y_ftr)

                # Predict
                y_te_pred = clf.predict(x_te['feats'])

                # Eval
                results_frame[leave_out][i_n][i_v] = accuracy_score(y_fte, y_te_pred)

            # Per sequence representation
            print '             Classificaton with sequence features'
            for i_n, (x_tr, x_te) in enumerate(zip(feat_X_str, feat_X_ste)):
                # Change classes
                #(x_tr, y_tr) = change_classes(x_tr['feats'], y_ftr, mode='6classes_true')
                #(x_te, y_te) = change_classes(x_te['feats'], y_fte, mode='6classes_true')

                # Train
                clf.fit(np.asarray(x_tr['feats']), y_str)

                # Predict
                y_te_pred = clf.predict(np.asarray(x_te['feats']))

                # Eval
                results_seq[leave_out][i_n][i_v] = accuracy_score(y_ste, y_te_pred)

    print np.mean(results_frame, axis = 0)
    print np.mean(results_seq, axis = 0)

if __name__ == '__main__':
    full_bow('/Users/cipriancorneanu/Research/data/fake_emotions/sift/', 'femo_sift_small.pkl')

