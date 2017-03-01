__author__ = 'cipriancorneanu'

import cPickle
from data import FakeEmo
from full_bow import *
from  sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

# Load data
path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
data = cPickle.load(open(path_sift+'femo_sift_sem_1_20', 'rb'))[:10]

# Prepare data
femo = FakeEmo('')
X, y = femo.prepare_frames(data)

# Shape as per frame instance
X = np.asarray(X)
X = np.reshape(X, (X.shape[0], -1))

# Select true classes only
X, y = change_classes(X, y, mode = '12classes')
X, y = np.asarray(X), np.asarray(y)

# Average

# Reduce dimensionality
pca = PCA(n_components=0.95, whiten=True).fit(X[::5])
X = pca.transform(X)

# Define classifier
clf = LinearSVC()

n_repetitions = 5
accuracy = []
for i in range(0, n_repetitions):
    (X_tr, X_te, y_tr, y_te)  = train_test_split(X, y, test_size=0.2)

    '''
    pool = np.random.choice(len(X_tr), pool_length, replace=False)
    X_tr = X_tr[pool]
    y_tr = y_tr[pool]
    '''

    # Train
    clf.fit(X_tr, y_tr)

    # Predict
    y_te_pred = clf.predict(X_te)

    # Eval
    accuracy.append(accuracy_score(y_te, y_te_pred))
    print 'Accuracy={}'.format(accuracy_score(y_te, y_te_pred))
    print confusion_matrix(y_te, y_te_pred)

print 'accuracy: {}'.format(np.mean(accuracy))