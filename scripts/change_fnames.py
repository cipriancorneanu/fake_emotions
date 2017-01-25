__author__ = 'cipriancorneanu'

import os

path = '/data/hupba2/Datasets/FakefaceDataProc/Extracted_faces'

for dir in os.listdir(path):
    for emo in os.listdir(os.path.join(path,dir)):
        for f in os.listdir(os.path.join(path,dir,emo)):
            if f.endswith('.png'):
                if f.startswith('frame'):
                    root = f.split('.')[0][5:]
                elif '[' in f:
                    root = f.split('[')[0]
                else:
                    root = f.split('.')

                ext = '0'*(5 - len(root))

                print os.path.join(path,dir,emo,ext+root)

                #os.rename(f, ext+root)

