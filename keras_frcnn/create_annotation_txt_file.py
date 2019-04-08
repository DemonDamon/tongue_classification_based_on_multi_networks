# This file is for creating annotation.txt file using tongue_train.csv
# file, which contains 8 fields including 'filename','width','height',
# 'class','xmin','ymin','xmax','ymax'.

import pandas as pd
from optparse import OptionParser
import os
parser = OptionParser()

parser.add_option("-t", "--type", dest="data_set_type", help="Train or test dataset.")
parser.add_option("-p", "--csv_path", dest="csv_file_path", help="Path to CSV label file.")
parser.add_option("-o", "--output_path", dest="txt_output_path", help="Path to Output annotation.txt file.")

(options, args) = parser.parse_args()

if not options.csv_file_path:   # if filename is not given
	parser.error('Error: path to CSV label file must be specified. Pass --path to command line')

train = pd.read_csv(options.csv_file_path)
data = pd.DataFrame()
data['format'] = train['filename']
save_name_root = '/'.join(options.csv_file_path.split('/')[:-1])

# as the images are in train_images folder, add train_images before the image name
if options.data_set_type == 'train':
    for i in range(data.shape[0]):
        data['format'][i] = os.path.join(save_name_root, 'train/') + data['format'][i]
    save_name = os.path.join(options.txt_output_path,'annotate_train.txt')
elif options.data_set_type == 'test':
    for i in range(data.shape[0]):
        data['format'][i] = os.path.join(save_name_root, 'test/') + data['format'][i]
    save_name = os.path.join(options.txt_output_path,'annotate_test.txt')
else:
    raise ValueError("Please designate the type of dataset, 'train' or 'test'.")

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) \
                        + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class'][i]

data.to_csv(save_name, header=None, index=None, sep=' ')