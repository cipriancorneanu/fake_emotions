__author__ = 'cipriancorneanu'

import cPickle
from data import FakeEmo
from full_bow import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

# Load data
path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
data = cPickle.load(open(path_sift+'femo_sift_sem.pkl', 'rb'))

# Prepare data
femo = FakeEmo('')
X, y = femo.prepare_sequences(data, format='sift_sequences_per_frame')

# Select true classes only
X, y = change_classes(X, y, mode = '12classes')
X, y = np.asarray(X), np.asarray(y)

# Reduce dimensionality
pca = PCA(n_components=16, whiten=True).fit(np.concatenate(X)[::5])
X = [pca.transform(x) for x in X]

# Sumarize sequences
n_labels = 20
kmeans = [KMeans(n_clusters=n_labels, random_state=0).fit(x) for x in X]
label_space = [km.predict(x) for (x,km) in zip(X, kmeans)]
distance_space = [km.transform(x) for x,km in zip(X,kmeans)]

keyframe = np.zeros((len(label_space), n_labels))
for i,(ls, ds, x) in enumerate(zip(label_space, distance_space, X)):
    for label in range(0,n_labels):
        segment = np.where(ls==label)[0]
        keyframe[i, label] = segment[np.argmin(ds[np.where(ls==label), label])]

X = np.reshape(np.asarray([[x[i] for i in k] for k, x in zip(keyframe, X)]), (len(X), -1))

X = np.reshape(X, (57*20, -1))
y = np.concatenate([20*[x] for x in y])

# Define classifier
clf = LinearSVC()

n_repetitions = 5
accuracy = []

for i in range(0, n_repetitions):
    (X_tr, X_te, y_tr, y_te)  = train_test_split(X, y, test_size=0.2)

    # Train
    clf.fit(X_tr, y_tr)

    # Predict
    y_te_pred = clf.predict(X_te)

    # Eval
    accuracy.append(accuracy_score(y_te, y_te_pred))
    print 'Accuracy={}'.format(accuracy_score(y_te, y_te_pred))
    print confusion_matrix(y_te, y_te_pred)

print np.mean(accuracy)