import os  
import glob  
import pandas as pd  
import xml.etree.ElementTree as ET

data = []
with open('./model_data/tongue_classes.txt','r') as f:
    for row in f:
        data.append(row)

def xml_to_csv(path):  
    xml_list = []  
    for xml_file in glob.glob(path + '/*.xml'):  
        tree = ET.parse(xml_file)  
        root = tree.getroot()  
        for member in root.findall('object'):  
            value = (root.find('filename').text,  
                     int(root.find('size')[0].text),  
                     int(root.find('size')[1].text),  
                     member[0].text,
                     data.index(member[0].text),
                     int(member[4][0].text),  
                     int(member[4][1].text),  
                     int(member[4][2].text),  
                     int(member[4][3].text)  
                     )  
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'classid', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)  
    return xml_df  
  
  
def run(images_path, save_csv_path):  
    os.chdir(images_path)   
    xml_df = xml_to_csv(images_path)  
    xml_df.to_csv(save_csv_path, index=None)  
    print('Converted xml to csv successfully .')  

if __name__ == '__main__':
    run('/root/damon_files/data/holistic_tongue_data/train','/root/damon_files/tongue_classification_based_on_multi_networks/keras_yolo3/tongue_train.csv')
    run('/root/damon_files/data/holistic_tongue_data/test','/root/damon_files/tongue_classification_based_on_multi_networks/keras_yolo3/tongue_test.csv')
