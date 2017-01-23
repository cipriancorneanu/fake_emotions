__author__ = 'cipriancorneanu'

import os

[os.rename(f, 'frame'+f) for f in os.listdir('.') if f.endsiwith('.png')]

