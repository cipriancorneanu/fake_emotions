__author__ = 'cipriancorneanu'

import os


path = '/Users/cipriancorneanu/Research/data/fake_emotions_proc/'
path_results = '/Users/cipriancorneanu/Research/data/fake_emotions_proc_res/'

for person in range(16,17):
    print('Copying files to local ...')

    # Copy from remote
    #os.system('scp -rq server:/data/hupba2/Datasets/FakefaceDataProc/Extracted_faces/' + str(person) + ' ' + path)

    # Run matlab code
    os.system("/Applications/MATLAB_R2014b.app/bin/matlab -nodisplay -nosplash -nodesktop -r 'read; exit;' ")

    # Copy result to remote
    os.system('scp ' + path_results + str(person) + '.mat server:')

    # Delete result
    os.system('rm ' + path_results + str(person) + '.mat')

    # Delete
    os.system('rm -rf ' + path + str(person))

