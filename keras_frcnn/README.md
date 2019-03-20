# keras-frcnn
Keras implementation of Faster R-CNN on Tongue Detection: Towards Real-Time Object Detection with Region Proposal Networks.
cloned from https://github.com/kbardool/keras-frcnn which using VGG-16 or ResNet-50 as backbone.

USAGE:
- Use tensorflow as backend is highly recommended.
- I prefer using LableImg(you can obtain from: https://tzutalin.github.io/labelImg/) to label, and it will generate .xml file for 
each image, then using `xml_to_csv.py` to unify the main information like `filename,width,height,class,xmin,ymin,xmax,ymax` of each 
picture into one .csv file.
- Afterwards, create .txt file by `create_annotation_txt_file.py` for training convenience, with each line containing:

    `filepath,x1,y1,x2,y2,class_name`
    
    For example:

    ./test_image/WX00852.jpg,577,635,1753,2080,tongue
    
    ./test_image/WX00853.jpg,842,628,1818,2114,tongue
    
- `train_frcnn.py` can be used to train a model. To train on self-defined dataset(if .txt file created by prior step named `annotate.txt`,
simply do: 

    `python train_frcnn.py --path ./annotate.txt`
    
    But if want to utilize augmentation and transfer learning, we can set more complicated like:

    ` python train_frcnn.py --path annotate.txt --parser self-defined --num_epochs 100 --horizontal_flips True 
--vertical_flips True --rot_90 True --utilize_transfer_learning True --including_top_weight False --output_weight_path './model_name.hdf5'`

    `train_frcnn.py` Parameter explanation:
    
    `--path`: Path of training data

    `--parser`: Parser to use. One of self-defined or pascal_voc, and default is self-defined
    
    `--num_epochs`: Number of epochs
    
    `--num_rois`: Number of RoIs to process at once

    `--backbone_network`: Base network including vgg and resnet50, and default is self-defined
    
    `--horizontal_flips`: Augment with horizontal flips in training. (Default=false)
    
    `--vertical_flips`: Augment with vertical flips in training. (Default=false)
    
    `--rot_90`: Augment with 90 degree rotations in training. (Default=false)
    
    `--utilize_transfer_learning`: Augment with using pre-trained model. (Default=false)
    
    `--including_top_weight`: Augment with including top weight which between input and second layer network. (Default=false)
    
    `--input_pretrained_weight_path`: Input path of pre-trained weights
    
    `--output_weight_path`: Output path for weights
    
    `--config_filename`: Location to store all the metadata related to the training (to be used when testing)

- Running `train_frcnn.py` will write weights to disk to .hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded by `test_frcnn.py` for any testing.

- `test_frcnn.py` can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images:
    ` python test_frcnn.py python test_frcnn.py --path ./test_images --input_weight_path './model_name.hdf5' 
    --is_save_cropped True --is_save_whole_image True --save_img_root_folder_path './test_predicted_output'`
    
    `test_frcnn.py` Parameter explanation:
    
    `--path`: Path of test data

    `--num_rois`: Number of RoIs to process at once, default=32
    
    `--backbone_network`: Base network including vgg and resnet50, default=resnet50
    
    `--config_filename`: Location to read the metadata related to the training (generated when training), default=config.pickle
    
    `--input_weight_path`: Input path for trained weights
    
    `--is_show_cropped`: Whether plot the cropped images. (Default=false)
    
    `--is_save_cropped`: Whether save the cropped images. (Default=false)
    
    `--is_show_whole`: Whether plot the whole images with predicted bounding box. (Default=false)
    
    `--is_save_whole_image`: Whether save the whole images with predicted bounding box. (Default=false)
    
    `--save_img_root_folder_path`: Folder path of storing predicted images


NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
- The theano backend by default uses a 7x7 pooling region, instead of 14x14 as in the frcnn paper. This cuts down compiling time slightly.
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is much more efficient and has little impact on results.


Example output:

![ex1](http://i.imgur.com/7Lmb2RC.png)
![ex2](http://i.imgur.com/h58kCIV.png)
![ex3](http://i.imgur.com/EbvGBaG.png)
![ex4](http://i.imgur.com/i5UAgLb.png)

ISSUES:

- If you get this error:
`ValueError: There is a negative shape in the graph!`    
    than update keras to the newest version

- This repo was developed using `python2`. `python3` should work thanks to the contribution of a number of users.

- If you run out of memory, try reducing the number of ROIs that are processed simultaneously. Try passing a lower `-n` to `train_frcnn.py`. Alternatively, try reducing the image size from the default value of 600 (this setting is found in `config.py`.
