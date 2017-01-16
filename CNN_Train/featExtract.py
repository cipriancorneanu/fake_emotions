# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/kulkarni/softs/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


caffe.set_device(2)
caffe.set_mode_cpu()

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
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227


#INPUT FACE IMAGE HERE

image = caffe.io.load_image(caffe_root + 'examples/images/S005_001_00000002.png')
transformed_image = transformer.preprocess('data', image)


# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

#EXAMPLE OF EXTRACTING FEAT FROM POOL5 LAYER
feat = net.blobs['pool5'].data[0]
