__author__ = 'cipriancorneanu'

import cPickle
from data import FakeEmo
from full_bow import *
from  sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

lebrow, rebrow, leye, reye, nose, mouth = ([1,3,6,7], [0,2,4,5], [9,11,14,15,17],
                                           [8,10,12,13,16], range(18,21), range(22,28))

grouping = [lebrow + leye, rebrow + reye, nose, mouth]
n_bins = [8, 8, 4, 8]
n_persons = 9

# Load data
path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/vgg/'
#data = []

#for i in range(0,7):
data = cPickle.load(open(path_sift+'femo_vgg_fc7_0.pkl', 'rb'))

# Prepare data
femo = FakeEmo('')

clf = LinearSVC()
results = np.zeros(n_persons)

for leave_out in range(0, n_persons):
    print 'Leave {} out'.format(leave_out)

    # Prepare sequences
    (X_tr, y_tr), (X_te, y_te) = femo.leave_one_out(data, leave_out, format='vgg_sequences')

    # Pool equally spaces frames
    X_tr_pl = [x[np.linspace(0.1*len(x), 0.9*len(x), num=5, dtype=np.int16)] for x in X_tr]
    X_te_pl = [x[np.linspace(0.1*len(x), 0.9*len(x), num=5, dtype=np.int16)] for x in X_te]

    # Reduce dimensionality
    X_pca = np.concatenate(X_tr)
    pca = PCA(n_components=0.5, whiten=True).fit(X_pca[::5])

    X_tr = np.asarray([np.concatenate(x) for x in [pca.transform(x) for x in X_tr_pl]])
    X_te = np.asarray([np.concatenate(x) for x in [pca.transform(x) for x in X_te_pl]])

    print X_tr.shape
    
    # Per sequence representation
    print '             Classificaton with sequence features'
    clf.fit(X_tr, y_tr)

    # Predict
    y_te_pred = clf.predict(X_te)

    # Eval
    results[leave_out] = accuracy_score(y_te, y_te_pred)
    print 'accuracy={}'.format(accuracy_score(y_te, y_te_pred))
    print confusion_matrix(y_te, y_te_pred)

print np.mean(results)