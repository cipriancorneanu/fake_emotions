__author__ = 'cipriancorneanu'

import cPickle
from data import FakeEmo
from full_bow import *
from  sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

'''
data  = cPickle.load(open('/Users/cipriancorneanu/Research/data/fake_emotions/geoms/femo_geom_raw.pkl', 'rb'))
lm = data[0][0][0]
fig, ax = plt.subplots()
ax.scatter(lm[:,0], lm[:,1])
for i, pt in enumerate(lm):
    ax.annotate(str(i), (pt[0], pt[1]))

plt.savefig('./landmarks.png')
'''

lebrow, rebrow, leye, reye, nose, mouth = ([1,3,6,7], [0,2,4,5], [9,11,14,15,17],
                                           [8,10,12,13,16], range(18,21), range(22,28))

grouping = [lebrow + leye, rebrow + reye, nose, mouth]
n_bins = [8, 8, 4, 8]
n_persons = 9

# Load data
path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
data = cPickle.load(open(path_sift+'femo_sift_sem_1_10.pkl', 'rb'))

# Prepare data
femo = FakeEmo('')
X, y = femo.prepare_sequences(data, format='sift_sequences_per_frame_lm')

clf = LinearSVC()
results = np.zeros(n_persons)

for leave_out in range(0, n_persons):
    print 'Leave {} out'.format(leave_out)

    # Prepare sequences
    (X_str, y_tr), (X_ste, y_te) = femo.leave_one_out(data, leave_out, format='sift_sequences_per_frame_lm')

    # Pre-cluster
    Xtrg = [[x[:,np.asarray(g),:] for x in X_str] for g in grouping]
    Xteg = [[x[:,np.asarray(g),:] for x in X_ste] for g in grouping]

    final_xtr, final_xte = ([], [])
    for i, (xtrg, xteg) in enumerate(zip(Xtrg, Xteg)):
        # Perform PCA per groups
        x_pca = np.concatenate(xtrg)
        x_pca = np.reshape(x_pca, (len(x_pca), -1))
        pca = PCA(n_components=0.9, whiten=True).fit(x_pca[::5])

        xtrg = [pca.transform(np.reshape(x, (x.shape[0], -1))) for x in xtrg]
        xteg = [pca.transform(np.reshape(x, (x.shape[0], -1))) for x in xteg]

        # k-means
        km = generate_kmeans(np.concatenate(xtrg), n_bins[i])

        final_xtr.append(np.asarray(generate_features(xtrg, km)))
        final_xte.append(np.asarray(generate_features(xteg, km)))


    # Per sequence representation
    print '             Classificaton with sequence features'
    clf.fit(np.concatenate(final_xtr, axis=1), y_tr)

    # Predict
    y_te_pred = clf.predict(np.concatenate(final_xte, axis=1))

    # Eval
    results[leave_out] = accuracy_score(y_te, y_te_pred)
    print 'accuracy={}'.format(accuracy_score(y_te, y_te_pred))
    print confusion_matrix(y_te, y_te_pred)

print np.mean(results)