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

def acc_per_frame(gt, est):
    return accuracy_score(gt, est)

def acc_per_video(gt, est, slices):
    gt_, est_ = ([], [])
    for slice in slices:
        gt_sliced =[gt[s] for s in slice]
        est_sliced =  [est[s] for s in slice]

        # Get class majority in estimation
        h = np.histogram(est_sliced, bins=range(0,13))[0]

        est_.append(np.where(h==max(h))[0][0])
        gt_.append(gt_sliced[0])

    return accuracy_score(gt_, est_)

def classify(path2data, path2save, n_clusters, partitions, down_sampling=1):
    # Init classes
    path = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/'
    femo = FakeEmo(path)
    clf = LinearSVC()

    # Leave-one-out
    acc_frame = np.zeros((femo.n_persons, len(n_clusters), len(partitions)))
    acc_video = np.zeros((femo.n_persons, len(n_clusters), len(partitions)))

    for leave in range(0,femo.n_persons):
        print 'Leave {}'.format(leave)
        for i_n, n in enumerate(n_clusters):
            print '     {} clusters'.format(n)

            dt = cPickle.load(open(path2data+str(leave)+'_'+str(n)+'.pkl', 'rb'))

            (X_tr, X_te, y_tr, y_te)= (dt['X_tr'], dt['X_te'],dt['y_tr'], dt['y_te'])

            slices_tr = middle_partition(slice(y_tr), partitions)
            slices_te = middle_partition(slice(y_te), partitions)

            for i_s,(slice_tr,slice_te) in enumerate(zip(slices_tr, slices_te)):
                # Slice
                X_tr_sliced = [X_tr[s] for s in np.asarray(np.concatenate(slice_tr), dtype=np.int64)]
                y_tr_sliced = [y_tr[s] for s in np.asarray(np.concatenate(slice_tr), dtype=np.int64)]

                X_te_sliced = [X_te[s] for s in np.asarray(np.concatenate(slice_te), dtype=np.int64)]
                y_te_sliced = [y_te[s] for s in np.asarray(np.concatenate(slice_te), dtype=np.int64)]

                print X_tr.shape
                print X_te.shape

                # Train
                clf.fit(X_tr_sliced[::down_sampling], y_tr_sliced[::down_sampling])

                # Predict
                y_te_pred = clf.predict(X_te_sliced)

                # Save results
                cPickle.dump({'clf':clf, 'gt': y_te_sliced, 'est': y_te_pred, 'slice_tr': slice_tr, 'slice_te':slice_te},
                             open(path2save + str(leave) + '_' + str(n) + '_' + str(i_s) + '.pkl', 'wb'))

                # Evaluate
                acc_frame[leave, i_n, i_s] = acc_per_frame(y_te_sliced, y_te_pred)
                #acc_video[leave, i_n, i_s] = acc_per_video(y_te_sliced, y_te_pred, slice_te)

                print '         Slice {} AccuracyFrame={}'.format(i_s, acc_frame[leave, i_n, i_s])
                #print '         Slice {} AccuracyVideo={}'.format(i_s, acc_video[leave, i_n, i_s])

def run_classify(argv):
    opts, args = getopt.getopt(argv, '')
    (path2data, path2save, n_clusters, partitions, ds) = \
        (
            args[0], args[1], [int(x) for x in args[2].split(',')], [float(x) for x in args[3].split(',')], int(args[4])
        )

    classify(path2data, path2save, n_clusters, partitions, ds)

if __name__ == '__main__':
    classify(
        '/Users/cipriancorneanu/Research/data/fake_emotions/sift/',
        '/Users/cipriancorneanu/Research/data/fake_emotions/sift/',
        [50,100,200],
        [0.2,0.3,0.5],
        10
    )
    #run_classify(sys.argv[1:])