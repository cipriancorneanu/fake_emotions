__author__ = 'cipriancorneanu'

from sklearn.metrics import accuracy_score
import numpy as np
from data import FakeEmo
import cPickle

def acc_per_frame(gt, est):
    return accuracy_score(gt, est)

def acc_per_video(gt, est, slices):
    gt_, est_, offset = ([], [], 0)
    for slice in slices:
        if len(slice)>0:
            slice = np.asarray(slice) - slice[0] + offset
            offset += len(slice)

            # Slice
            gt_sliced =[gt[s] for s in slice]
            est_sliced =  [est[s] for s in slice]

            # Get class majority in estimation
            h = np.histogram(est_sliced, bins=range(0,13))[0]

            # Append majority class per video
            est_.append(np.where(h==max(h))[0][0])
            gt_.append(gt_sliced[0])

    return accuracy_score(gt_, est_)

if __name__ == '__main__':
    path2data = '/Users/cipriancorneanu/Research/data/fake_emotions/sift/sift/'

    n_persons, n_clusters, n_partitions = (54,3,4)

    # Leave-one-out
    acc_frame = np.zeros((n_persons, n_clusters, n_partitions))
    acc_video = np.zeros((n_persons, n_clusters, n_partitions))

    for fr in [5,20]:
        for leave in range(0,n_persons):
            for i_n, n in enumerate([50,100,200]):
                for i_s in range(0,n_partitions):
                    dt = cPickle.load(open(path2data+str(leave)+'_'+str(n)+'_'+str(i_s)+'_'+ str(fr)+'.pkl', 'rb'))

                    # Evaluate
                    acc_frame[leave, i_n, i_s] = acc_per_frame(dt['gt'], dt['est'])
                    acc_video[leave, i_n, i_s] = acc_per_video(dt['gt'], dt['est'], dt['slice_te'])

        print 'Frame rate {}:1'.format(fr)
        print np.mean(acc_frame, axis=0)
        print np.mean(acc_video, axis=0)