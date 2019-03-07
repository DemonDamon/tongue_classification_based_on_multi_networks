from __future__ import division

from random import shuffle, choice
from time import time
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
parser.add_option("--horizontal_flips", type="bool", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vertical_flips", type="bool", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot_90", type="bool", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",action="store_true", default=False)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",default="config.pickle")
parser.add_option("--utilize_transfer_learning", type="bool", dest="utilize_transfer_learning", help="Augment with using pre-trained model. (Default=true).",  action="store_true", default=True)
parser.add_option("--including_top_weight", type="bool", dest="including_top_weight", help="Augment with including top weight which between input and second layer network. (Default=false).",  action="store_true", default=False)
parser.add_option("--input_pretrained_weight_path", dest="input_pretrained_weight_path", help="Input path of pre-trained weights.")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')

(options, args) = parser.parse_args()

print('/n')
print(" " + "-"*30 + " Preparing Configuration and Data " + "-"*30)

if not options.train_path:   # if filename is not given
	parser.error(' [Error]: path of training data must be specified.')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'self-defined':
	from keras_frcnn.self_defined_parser import get_data
else:
	raise ValueError(" [ValueError]: Command line option parser must be one of 'pascal_voc' or 'self-defined'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.num_epochs = int(options.num_epochs)
C.num_rois = int(options.num_rois)
C.use_horizontal_flips = options.use_horizontal_flips == 'True'
C.use_vertical_flips = options.use_vertical_flips == 'True'
C.rot_90 = options.rot_90 == 'True'
C.utilize_transfer_learning = options.utilize_transfer_learning == 'True'
C.including_top_weight = options.including_top_weight == 'True'
# C.use_horizontal_flips = bool(options.horizontal_flips)
# C.use_vertical_flips = bool(options.vertical_flips)
# C.rot_90 = bool(options.rot_90)
# C.utilize_transfer_learning = bool(options.utilize_transfer_learning)
# C.including_top_weight = bool(options.including_top_weight)
C.model_path = options.output_weight_path

if options.backbone_network == 'vgg':
	C.backbone_network = 'vgg'
	from keras_frcnn import vgg as nn
elif options.backbone_network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.backbone_network = 'resnet50'
else:
	print(' [*] Not a valid model')
	raise ValueError

if options.utilize_transfer_learning and not options.input_pretrained_weight_path:
	# define to utilize transfer learning but if not specify "input_pretrained_weight_path" parameter,
	# then default to download models from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
	# to ./pretrained_model_weights folder.
	if not path.exists('./pretrained_model_weights'):
		C.base_net_weights = nn.download_imagenet_weight_file(options.including_top_weight)
	else:
		print(options.including_top_weight)
		if not options.including_top_weight:
			print(1)
			if path.exists('./pretrained_model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'):
				C.base_net_weights = './pretrained_model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
			else:
				C.base_net_weights = nn.download_imagenet_weight_file(options.including_top_weight)
		else:
			print(2)
			if path.exists('./pretrained_model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'):
				C.base_net_weights = './pretrained_model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
			else:
				C.base_net_weights = nn.download_imagenet_weight_file(options.including_top_weight)
elif options.input_pretrained_weight_path:
	# utilize self-defined pre-trained model weight, not download automatically but manually download
	# in ./pretrained_model_weights folder
	C.base_net_weights = options.input_pretrained_weight_path
else:
	# not use transfer learning
	C.base_net_weights = None


all_imgs, classes_count, class_mapping = get_data(options.train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print(' [*] Training images per class:', classes_count)
print(' [*] Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	dump(C,config_f)
	print(' [*] Print out all attributes of Config:')
	for i in vars(C):
		print('    ', i, ':', vars(C)[i])
	print(' [*] Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print(' [*] Num train samples {}'.format(len(train_imgs)))
print(' [*] Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')
print(' [*] Have created data generator of both train and test dataset. ')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)
print(' [*] Have created the holistic model well. ')

try:
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
	print(' [*] Have loaded weights from {}'.format(C.base_net_weights))
except:
	print(' [*] Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
print(' [*] Have set up the optimized method parameter. \n')

print(" " + "-"*30 + " Starting training " + "-"*30)

epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

start_time = time()
best_loss = np.Inf
class_mapping_inv = {v: k for k, v in class_mapping.items()}

vis = True

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print(' [*] Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('  - Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('  - RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)

			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []

			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if C.num_rois > 1:
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = choice(neg_samples)
				else:
					sel_samples = choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('  - Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('  - Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('  - Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('  - Loss RPN regression: {}'.format(loss_rpn_regr))
					print('  - Loss Detector classifier: {}'.format(loss_class_cls))
					print('  - Loss Detector regression: {}'.format(loss_class_regr))
					print('  - Elapsed time: {}'.format(time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time()

				if curr_loss < best_loss:
					if C.verbose:
						print('  - Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			print(' [*] Exception: {}'.format(e))
			continue

print(' [*] Training complete, exiting.')
