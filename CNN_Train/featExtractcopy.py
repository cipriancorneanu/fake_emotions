# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/kulkarni/softs/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


caffe.set_device(2)
caffe.set_mode_gpu()

model_def = '/home/kulkarni/Vision/faces/vgg_face_caffe/VGG_FACE_deploy.prototxt'
model_weights = '/home/kulkarni/Vision/faces/vgg_face_caffe/VGG_FACE.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#ENTER IMAGE MEANS HERE

mu = np.array([129.1863,104.7624,93.5940] )



# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227


#INPUT FACE IMAGE HERE


for root, dirs, filenames in os.walk("/data/hupba2/Datasets/Cohn-Kanade/cohn-kanade-images"):
    for f in filenames:
        filepath =  os.path.join(root, f)
        ext = os.path.splitext(filepath)[-1].lower()
        print filepath
        if ext == '.png':
           image = caffe.io.load_image(os.path.join(root, f))
           transformed_image = transformer.preprocess('data', image)
           # copy the image data into the memory allocated for the net
           net.blobs['data'].data[...] = transformed_image
           ### perform classification
           output = net.forward()
           feat = net.blobs['fc7'].data[0]
           (files, ext)=  os.path.splitext(f)
           filep = "/data/hupba2/Derived/CNNFacefeat/VGGOrigFC7/"+ files+".pkl"        
           pickle.dump( feat, open( filep, "wb" ) )

# for each layer, show the output shape
#for layer_name, blob in net.blobs.iteritems():
#   print layer_name + '\t' + str(blob.data.shape)

#EXAMPLE OF EXTRACTING FEAT FROM POOL5 LAYE
#feat = net.blobs['fc7'].data[0]
#print(np.nonzero(feat))
#print(feat[4075])
#print(feat.flatten())

