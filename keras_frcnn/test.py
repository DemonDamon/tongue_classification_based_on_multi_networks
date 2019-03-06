from __future__ import division

from random import shuffle, choice
from time import time
from pprint import pprint
from sys import setrecursionlimit
from os import path
from pickle import dump
import numpy as np
from optparse import OptionParser

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--path", dest="train_path", help="Path of training data.")
parser.add_option("--parser", dest="parser", help="Parser to use. One of self-defined or pascal_voc",default="pascal_voc")
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--backbone_network", dest="backbone_network", help="Base network including vgg and resnet50.", default='resnet50')
parser.add_option("--horizontal_flips", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vertical_flips", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",action="store_true", default=False)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",default="config.pickle")
parser.add_option("--utilize_transfer_learning", dest="utilize_transfer_learning", help="Augment with using pre-trained model. (Default=true).",  action="store_true", default=True)
parser.add_option("--including_top_weight", dest="including_top_weight", help="Augment with including top weight which between input and second layer network. (Default=false).",  action="store_true", default=False)
parser.add_option("--input_pretrained_weight_path", dest="input_pretrained_weight_path", help="Input path of pre-trained weights.")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
	parser.error('Error: path of training data must be specified.')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'self-defined':
	from keras_frcnn.self_defined_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'self-defined'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()




all_imgs, classes_count, class_mapping = get_data(options.train_path)

print(all_imgs)