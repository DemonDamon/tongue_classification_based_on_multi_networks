# This file is for creating annotation.txt file using tongue_train.csv
# file, which contains 8 fields including 'filename','width','height',
# 'class','xmin','ymin','xmax','ymax'.

import pandas as pd
from optparse import OptionParser
import os
parser = OptionParser()

parser.add_option("-p", "--csv_path", dest="csv_file_path", help="Path to CSV label file.")
parser.add_option("-o", "--output_path", dest="txt_output_path", help="Path to Output annotation.txt file.")

(options, args) = parser.parse_args()

if not options.csv_file_path:   # if filename is not given
	parser.error('Error: path to CSV label file must be specified. Pass --path to command line')

train = pd.read_csv(options.csv_file_path)
data = pd.DataFrame()
data['format'] = train['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = '/root/damon_files/data/holistic_tongue_data/train/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    # data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) \
    #                     + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class'][i]
    data['format'][i] = data['format'][i] + ' ' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) \
                        + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + str(train['classid'][i])

with open(os.path.join(options.txt_output_path,'annotate.txt'), 'w') as f:
    for i in data['format']:
        f.writelines(i+'\n')
# data.to_csv(os.path.join(options.txt_output_path,'annotate.txt'), header=None, index=None, sep=' ')