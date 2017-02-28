__author__ = 'cipriancorneanu'

import cPickle
from data import FakeEmo
from full_bow import *
from  sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

n_persons = 5
n_clusters = [50, 100, 200]

# Load data
path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
data = cPickle.load(open(path_sift+'femo_sift_sem.pkl', 'rb'))

# Prepare data
femo = FakeEmo('')
X, y = femo.prepare_sequences(data, format='sift_sequences')

clf = LinearSVC()
n_clusters = [256]
results = np.zeros((n_persons, len(n_clusters)))

for leave_out in range(0, n_persons):
    print 'Leave {} out'.format(leave_out)

    # Prepare sequences
    (X_str, y_str), (X_ste, y_ste) = femo.leave_one_out(data, leave_out, format='sift_sequences')

    # Perform PCA
    pca = PCA(n_components=0.95, whiten=True).fit(np.concatenate(X_str)[::5])
    X_str_pca = [pca.transform(x) for x in X_str]
    X_ste_pca = [pca.transform(x) for x in X_ste]

    # k-means
    (_, feat_X_str, feat_X_ste) = generate_all(X_str_pca, X_ste_pca, n_clusters)

    # Per sequence representation
    print '             Classificaton with sequence features'
    for i_n, (x_tr, x_te) in enumerate(zip(feat_X_str, feat_X_ste)):
        # Train
        clf.fit(np.asarray(x_tr['feats']), y_str)

        # Predict
        y_te_pred = clf.predict(np.asarray(x_te['feats']))

        # Eval
        results[leave_out][i_n] = accuracy_score(y_ste, y_te_pred)
        print confusion_matrix(y_ste, y_te_pred)

print results