import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", dest="xml_path", help="Path of folder saving .xml data.")

(options, args) = parser.parse_args()

path = options.xml_path

assert path != None

data = []; class_map = {}; class_count_dict = {}
with open('./tongue_classes.txt','r') as f:
    for id, row in enumerate(f):
        data.append(row)
        class_map.update({str(id+1):id})
        class_count_dict.update({str(id+1):0})

def xml_to_pkl():
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        value = {'filepath':os.path.join(path,root.find('filename').text),
                'width':int(root.find('size')[0].text),
                'height':int(root.find('size')[1].text),
                'imageset':'trainval'}
        bboxes = []
        for member in root.findall('object'):
            box = {'class' : int(data.index(member[0].text))+1,
                'x1' : int(member[4][0].text),
                'y1' : int(member[4][1].text),
                'x2' : int(member[4][2].text),
                'y2' : int(member[4][3].text)}
            bboxes.append(box)
            class_count_dict[int(data.index(member[0].text))+1] += 1
        value.update({'bboxes':bboxes})
        xml_list.append(value)
    data = (xml_list,class_count_dict,class_map)
    # column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    # xml_df = pd.DataFrame(xml_list, columns=column_name)
    pickle.dump(data,open('./data.pkl','wb'))


xml_to_pkl()
print('Converted xml to pkl successfully .')


# if __name__ == '__main__':
#    run('D:\\workfiles\\tongue_classification_based_on_multi_networks\\keras_frcnn\\test_image')
