import os  
import glob  
import pandas as pd  
import xml.etree.ElementTree as ET  
  
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
                     int(member[4][0].text),  
                     int(member[4][1].text),  
                     int(member[4][2].text),  
                     int(member[4][3].text)  
                     )  
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']  
    xml_df = pd.DataFrame(xml_list, columns=column_name)  
    return xml_df  
  
  
def run(images_path, save_csv_path):  
    os.chdir(images_path)   
    xml_df = xml_to_csv(images_path)  
    xml_df.to_csv(save_csv_path, index=None)  
    print('Converted xml to csv successfully .')  

if __name__ == '__main__':
    # run('D:\\workfiles\\tongue_project(MaskRCNN and TF Object Detectuion API)\\tonge_recognization_tf_object_detection_API\\data\\train',\
    #  'D:\\workfiles\\tongue_project(MaskRCNN and TF Object Detectuion API)\\tonge_recognization_tf_object_detection_API\\data\\tongue_train.csv')  
    # run('D:\\workfiles\\tongue_project(MaskRCNN and TF Object Detectuion API)\\tonge_recognization_tf_object_detection_API\\data\\test',\
    #  'D:\\workfiles\\tongue_project(MaskRCNN and TF Object Detectuion API)\\tonge_recognization_tf_object_detection_API\\data\\tongue_test.csv')  
    run('./data/train','./data/tongue_train.csv')  
    run('./data/test','./data/tongue_test.csv')  
 