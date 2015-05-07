import scipy
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os

# Important Note: This python programs is assumed to be located inside
# the $CAFFE_ROOT/examples directory. The Images are assumed to be located
# inside the $CAFFE_ROOT/test_imgs directory

caffe_root = '../'

# Set Path Variables
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGES_DIR = '../data/test_imgs'

# Open the text file containing the labels, and eliminate the synset labels
# beginning each line
text_file = open("../data/ilsvrc12/synset_words.txt", "r")
lines = text_file.readlines()
w = [line.split(' ', 1)[1] for line in lines]

caffe.set_mode_gpu()

# Load a pretrained network, subtracting the mean, swapping input channels
# to BGD instead of rgd, scaling the images to [0,255].
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
	mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
	channel_swap=(2,1,0), 
	raw_scale=255, 
	image_dims=(256,256))

print '\n------------------ Prediction using oversampling: ------------------'

# Loop over all the image files in the specified images directory
for subdir, _, files in os.walk(IMAGES_DIR):
	for file in files:
		print 'File name:', file
		input_image = caffe.io.load_image(os.path.join(subdir, file))
		
		# By enabling oversampling, caffe generates a batch of
		# 10 images, the center, corner crops as well as their
		# mirrors, and uses their average prediction.
		prediction = net.predict([input_image], oversample=True)

		print 'Predicted class index:', prediction[0].argmax(), 'corresponding to', w[prediction[0].argmax()],'predicted with a probability of', prediction[0].max()

		print 'Entropy:', scipy.stats.entropy(prediction[0], base=2), '\n'

print '\n------------------ Prediction without oversampling: ------------------'

# Loop over all the image files in the specified images directory
for subdir, _, files in os.walk(IMAGES_DIR):
	for file in files:
		print 'File name:', file
		input_image = caffe.io.load_image(os.path.join(subdir, file))
		
		# By disabling oversampling, caffe only uses one image
		# with the given orientation to obtain a class prediction
		prediction = net.predict([input_image], oversample=False)

		print 'Predicted class index:', prediction[0].argmax(), 'corresponding to', w[prediction[0].argmax()],'predicted with a probability of', prediction[0].max()

		print 'Entropy:', scipy.stats.entropy(prediction[0], base=2), '\n'
