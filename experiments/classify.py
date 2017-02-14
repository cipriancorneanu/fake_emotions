__author__ = 'cipriancorneanu'

from data import FakeEmo
import cPickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import getopt
import sys

def slice_sequences(data):
    lengths = [len(d) for d in data]
    return [range(0,l) for l in lengths]

def slice(labels):
    borders = [i+1 for i,(x, x_) in enumerate(zip(labels[:-1], labels[1:])) if x!=x_]
    return [range(bmin, bmax) for bmin, bmax in zip([0]+borders, borders + [len(labels)])]

def middle_partition(slices, partitions):
    middle_partitions = [None]*len(partitions)

    for i,part in enumerate(partitions):
         middle_partitions[i] = [s[int(len(s)*part/2):int(len(s)*(1-part/2))] for s in slices]

    return middle_partitions

def change_class_pair (X, y, label1, label2):
        t  = [ (v_X, v_y-label1) for (v_X, v_y) in zip(X, y) if v_y==label1 ]
        f  = [ (v_X, v_y-label2+1) for (v_X, v_y) in zip(X, y) if v_y==label2 ]
        X_t, y_t = zip(*t) if t else ([],[])
        X_f, y_f = zip(*f) if f else ([],[])
        if X_f and X_t:
            return (X_t+X_f, y_t+y_f)
        else:
            return ([],[])

def change_classes(X, y, mode='12classes'):
    if mode == '2classes_happy':
        return change_class_pair(X, y, 0, 6)
    elif mode == '2classes_sad':
        return change_class_pair(X, y, 1, 7)
    elif mode == '2classes_contempt':
        return change_class_pair(X, y, 2, 8)
    elif mode == '2classes_surprised':
        return change_class_pair(X, y, 3, 9)
    elif mode == '2classes_disgusted':
        return change_class_pair(X, y, 4, 10)
    elif mode == '2classes_angry':
        return change_class_pair(X, y, 5, 11)
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
    n_persons, n_partitions = (54, 1)
    clf = LinearSVC()

    print mode
    results = []#np.zeros((n_persons, len(n_clusters)))

    for leave in range(0, n_persons):
        for i_n, n in enumerate(n_clusters):
            for i_p in range(0,n_partitions):

                dt = cPickle.load(open(path2data+str(leave)+'_'+ str(i_p)+'_'+ str(n)+'.pkl', 'rb'))

                (X_tr, X_te, y_tr, y_te)= (dt['X_tr']['feats'], dt['X_te']['feats'], dt['y_tr'], dt['y_te'])

                # Change classes
                (X_tr_,y_tr_) = change_classes(X_tr, y_tr, mode)
                (X_te_,y_te_) = change_classes(X_te, y_te, mode)

                if X_tr_ and X_te_:
                    # Train
                    clf.fit(X_tr_, y_tr_)

                    # Predict
                    y_te_pred = clf.predict(X_te_)

                    # Eval
                    results.append(accuracy_score(y_te_, y_te_pred))

                    # Save results
                    '''
                    if save:
                        cPickle.dump({'clf':clf, 'gt': y_te, 'est': y_te_pred},
                                     open(path2save + str(leave)+'_'+ str(i_p)+'_'+ str(n) + '.pkl', 'wb'))
                    '''

    print np.mean(np.reshape(results, (-1,len(n_clusters))), axis=0)


def run_classify(argv):
    opts, args = getopt.getopt(argv, '')
    (path2data, path2save, n_clusters, partitions, ds) = \
        (
            args[0], args[1], [int(x) for x in args[2].split(',')], [float(x) for x in args[3].split(',')], int(args[4])
        )

    classify_frame(path2data, path2save, n_clusters, partitions, ds)

if __name__ == '__main__':
    path_sift =  '/Users/cipriancorneanu/Research/data/fake_emotions/vgg/'

    print 'Sequence classification'
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '12classes'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '6classes_true'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '6classes_fake'
    )
    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_happy'
    )

    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_sad'
    )

    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_contempt'
    )

    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_surprised'
    )

    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_disgusted'
    )

    classify_sequence(
        path_sift + 'bow_per_video/',
        path_sift + 'bow_per_video/',
        [50,100,200], '2classes_angry'
    )
    #run_classify(sys.argv[1:])