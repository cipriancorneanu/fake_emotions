__author__ = 'cipriancorneanu'

from data import FakeEmo
import cPickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import getopt
import sys

def slice(labels):
    borders = [i+1 for i,(x, x_) in enumerate(zip(labels[:-1], labels[1:])) if x!=x_]
    return [range(bmin, bmax) for bmin, bmax in zip([0]+borders, borders + [len(labels)])]

def middle_partition(slices, partitions):
    middle_partitions = [None]*len(partitions)

    for i,part in enumerate(partitions):
         middle_partitions[i] = [s[int(len(s)*part/2):int(len(s)*(1-part/2))] for s in slices]

    return middle_partitions

def change_classes(X, y, mode='12classes'):
    if mode == '2classes':
        return (X, [0 if x < 6 else 1 for x in y])
    elif mode == '6classes_true':
        return  zip(*[ (v_X, v_y) for (v_X, v_y) in zip(X, y) if v_y<6 ])
    elif mode == '6classes_fake':
        return  zip(*[ (v_X, v_y-6) for (v_X, v_y) in zip(X, y) if v_y>=6 ])
    elif mode == '12classes':
        return (X, y)

def classify_frame(path2data, path2save, n_clusters, partitions, down_sampling=1, mode='12classes', save=False):
    n_persons = 54
    clf = LinearSVC()

    # Leave-one-out
    results = np.zeros((n_persons, len(n_clusters), len(partitions)))

    for leave in range(0,n_persons):
        print 'Leave {}'.format(leave)
        for i_n, n in enumerate(n_clusters):
            print '     {} clusters'.format(n)

            dt = cPickle.load(open(path2data+str(leave)+'_'+str(n)+'.pkl', 'rb'))

            (X_tr, X_te, y_tr, y_te)= (dt['X_tr'], dt['X_te'],dt['y_tr'], dt['y_te'])

            # Change classes
            (X_tr,y_tr) = change_classes(X_tr, y_tr, mode)
            (X_te,y_te) = change_classes(X_te, y_te, mode)

            # Slice
            slices_tr = middle_partition(slice(y_tr), partitions)
            slices_te = middle_partition(slice(y_te), partitions)

            for i_s,(slice_tr,slice_te) in enumerate(zip(slices_tr, slices_te)):
                # Slice
                X_tr_sliced = [X_tr[s] for s in np.asarray(np.concatenate(slice_tr), dtype=np.int64)]
                y_tr_sliced = [y_tr[s] for s in np.asarray(np.concatenate(slice_tr), dtype=np.int64)]

                X_te_sliced = [X_te[s] for s in np.asarray(np.concatenate(slice_te), dtype=np.int64)]
                y_te_sliced = [y_te[s] for s in np.asarray(np.concatenate(slice_te), dtype=np.int64)]

                # Train
                clf.fit(X_tr_sliced[::down_sampling], y_tr_sliced[::down_sampling])

                # Predict
                y_te_pred = clf.predict(X_te_sliced)

                # Eval
                results[leave, i_n, i_s] = accuracy_score(y_te, y_te_pred)

                # Save results
                if save:
                    cPickle.dump({'clf':clf, 'gt': y_te_sliced, 'est': y_te_pred, 'slice_tr': slice_tr, 'slice_te':slice_te},
                                 open(path2save + str(leave) + '_' + str(n) + '_' + str(i_s) + '_' + str(down_sampling) + '.pkl', 'wb'))

    print np.mean(results, axis=0)


def classify_sequence(path2data, path2save, n_clusters, mode='12classes', save=False):
    n_persons, n_partitions = (54, 4)
    clf = LinearSVC()

    print mode
    results = np.zeros((n_persons, len(n_clusters)))

    for leave in range(0, n_persons):
        for i_n, n in enumerate(n_clusters):
            for i_p,(slice_tr,slice_te) in range(0,n_partitions):
                dt = cPickle.load(open(path2data+str(leave)+'_'+ str(i_p)+'_'+ str(n)+'.pkl', 'rb'))

                (X_tr, X_te, y_tr, y_te)= (dt['X_tr'], dt['X_te'],dt['y_tr'], dt['y_te'])

                # Normalize
                X_tr = [x/np.sum(x) for x in X_tr]
                X_te = [x/np.sum(x) for x in X_te]

                # Change classes
                (X_tr,y_tr) = change_classes(X_tr, y_tr, mode)
                (X_te,y_te) = change_classes(X_te, y_te, mode)

                # Train
                clf.fit(X_tr, y_tr)

                # Predict
                y_te_pred = clf.predict(X_te)

                # Eval
                results[leave, i_n] = accuracy_score(y_te, y_te_pred)

                # Save results
                if save:
                    cPickle.dump({'clf':clf, 'gt': y_te, 'est': y_te_pred},
                                 open(path2save + str(leave)+'_'+ str(i_p)+'_'+ str(n) + '.pkl', 'wb'))

    print np.mean(results, axis=0)


def run_classify(argv):
    opts, args = getopt.getopt(argv, '')
    (path2data, path2save, n_clusters, partitions, ds) = \
        (
            args[0], args[1], [int(x) for x in args[2].split(',')], [float(x) for x in args[3].split(',')], int(args[4])
        )

    classify_frame(path2data, path2save, n_clusters, partitions, ds)

if __name__ == '__main__':
    path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'

    print 'Sequence classification'
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'results_bow_per_video_12c/',
        [50,100,200], '12classes'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'results_bow_per_video_12c/',
        [50,100,200], '6classes_true'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'results_bow_per_video_12c/',
        [50,100,200], '6classes_fake'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'results_bow_per_video_12c/',
        [50,100,200], '2classes'
    )

    #run_classify(sys.argv[1:])