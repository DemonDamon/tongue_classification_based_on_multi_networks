from __future__ import division

import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils

from keras_frcnn import roi_helpers
import matplotlib.pylab as plt

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--path", dest="test_path", help="Path of test data.")
parser.add_option("--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--backbone_network", dest="backbone_network", help="Base network including vgg and resnet50.", default='resnet50')
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",default="config.pickle")
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for trained weights.", default='./model_frcnn.hdf5')
parser.add_option("--is_show_cropped", dest="is_show_cropped", help="Whether plot the cropped images. (Default=false)", default=False)
parser.add_option("--is_save_cropped", dest="is_save_cropped", help="Whether save the cropped images. (Default=false)", default=False)
parser.add_option("--is_show_whole", dest="is_show_whole", help="Whether plot the whole images with predicted bounding box. (Default=false)", default=False)
parser.add_option("--is_save_whole_image", dest="is_save_whole_image", help="Whether save the whole images with predicted bounding box. (Default=false)", default=False)
parser.add_option("--save_img_root_folder_path", dest="save_img_root_folder_path", help="Folder path of storing predicted images. ")

(options, args) = parser.parse_args()

print('\n')
print(" " + "-"*30 + " Preparing Configuration and Data " + "-"*30)

if not options.test_path:   # if filename is not given
	parser.error(' [Error]: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.backbone_network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.backbone_network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.backbone_network == 'resnet50':
	num_features = 1024
elif C.backbone_network == 'vgg':
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base backbone network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

try:
	if options.input_weight_path:
		C.model_path = options.input_weight_path
	print(' [*] Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)
except:
	print(' [Error]: Could not load trained model weights.')

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []
classes = {}
bbox_threshold = 0.8
visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	progbar = generic_utils.Progbar(len(os.listdir(img_path)))

	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	# print(' [*]' + img_name + ' is done. ')
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []
	figID = 0
	for key in bboxes:
		bbox = np.array(bboxes[key])
		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			tmpImg = img[real_y1:real_y2, real_x1:real_x2]
			if tmpImg.shape[0] != 0 and tmpImg.shape[1] != 0:
				b, g, r = cv2.split(tmpImg)
				tmpImg = cv2.merge([r, g, b])

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

			if options.is_show_cropped:
				plt.figure(figID, figsize=(30, 20))
				figID += 1
				plt.title(textLabel)
				plt.imshow(tmpImg)

			if options.is_save_cropped:
				if options.save_img_root_folder_path:
					save_path = os.path.join(options.save_img_root_folder_path,img_name.split('.')[0])
				else:
					save_path = os.path.join(img_path,img_name.split('.')[0])
				if not os.path.exists(save_path):
					os.mkdir(save_path)
				if new_boxes.shape[0] == 1:
					tongue_cropped_whole_path = os.path.join(save_path, img_name.split('.')[0] + '_whole.jpg')
					shebian_1_path = os.path.join(save_path, img_name.split('.')[0] + '_shebian_1.jpg')
					shebian_2_path = os.path.join(save_path, img_name.split('.')[0] + '_shebian_2.jpg')
					shegen_path = os.path.join(save_path, img_name.split('.')[0] + '_shegen.jpg')
					shejian_path = os.path.join(save_path, img_name.split('.')[0] + '_shejian.jpg')
					shezhong_path = os.path.join(save_path, img_name.split('.')[0] + '_shezhong.jpg')

				else:
					tongue_cropped_whole_path = os.path.join(save_path, img_name.split('.')[0] + '_whole_' + str(jk+1) + '.jpg')
					shebian_1_path = os.path.join(save_path, img_name.split('.')[0] + '_shebian_1_' + str(jk+1) + '.jpg')
					shebian_2_path = os.path.join(save_path, img_name.split('.')[0] + '_shebian_2_' + str(jk+1) + '.jpg')
					shegen_path = os.path.join(save_path, img_name.split('.')[0] + '_shegen_' + str(jk+1) + '.jpg')
					shejian_path = os.path.join(save_path, img_name.split('.')[0] + '_shejian' + str(jk+1) + '.jpg')
					shezhong_path = os.path.join(save_path, img_name.split('.')[0] + '_shezhong' + str(jk+1) + '.jpg')

				b, g, r = cv2.split(tmpImg)
				tmpImg = cv2.merge([r, g, b])
				cv2.imwrite(tongue_cropped_whole_path, tmpImg)

				img_height, img_width = tmpImg.shape[:2]

				shebian_1 = tmpImg[:, :int(img_width / 5)]
				cv2.imwrite(shebian_1_path, shebian_1)

				shebian_2 = tmpImg[:, -int(img_width / 5):]
				cv2.imwrite(shebian_2_path, shebian_2)

				left = tmpImg[:, int(img_width / 5):img_width - int(img_width / 5)]

				shegen = left[:int(img_height / 3), :]
				cv2.imwrite(shegen_path, shegen)

				shejian = left[-int(img_height / 6):, :]
				cv2.imwrite(shejian_path, shejian)

				shezhong = left[int(img_height / 3):img_height - int(img_height / 6), :]
				cv2.imwrite(shezhong_path, shezhong)

				# plt.figure(idx+200, figsize=(30, 20))
				# plt.subplot(3, 4, 3);plt.imshow(shegen);plt.title('tongue_root')
				# plt.subplot(3, 4, 5);plt.imshow(img);plt.title('cropped_tongue')
				# plt.subplot(3, 4, 6);plt.imshow(shebian_1);plt.title('tongue_margin_left')
				# plt.subplot(3, 4, 7);plt.imshow(shezhong);plt.title('tongue_middle')
				# plt.subplot(3, 4, 8);plt.imshow(shebian_2);plt.title('tongue_margin_right')
				# plt.subplot(3, 4, 11);plt.imshow(shejian);plt.title('tongue_tip')
				# plt.savefig(os.path.join(save_path, img_name.split('.')[0] + '_divided.jpg'))

	[r, g, b] = cv2.split(img)
	new_img = cv2.merge([b, g, r])

	if options.is_show_whole:
		plt.figure(idx + 100, figsize=(30, 20))
		plt.imshow(new_img)

	if options.is_save_whole_image:
		if options.save_img_root_folder_path:
			save_path = os.path.join(options.save_img_root_folder_path, img_name.split('.')[0])
		else:
			save_path = os.path.join(img_path, img_name.split('.')[0])
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		[r, g, b] = cv2.split(new_img)
		new_img = cv2.merge([b, g, r])
		cv2.imwrite(os.path.join(save_path, img_name), new_img)

	progbar.update(idx+1)

