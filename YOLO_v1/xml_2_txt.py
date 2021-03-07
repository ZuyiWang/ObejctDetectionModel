import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            # print(filename)  
            # difficult表明这个待检测目标很难识别，有可能是虽然视觉上很清楚，但是没有上下文的话还是
            # 很难确认它属于哪个分类； 标为difficult的目标在测试成绩的评估中一般会被忽略。
            continue
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

# txt_file = open('voc2007test.txt','w')
txt_file = open('voc2012_tra_val.txt','w')
# test_file = open('voc07testimg.txt','r')
# lines = test_file.readlines()
# lines = [x[:-1] for x in lines]
# print(lines)

# Annotations = '/media/iaes/新加卷/wangzy/VOC07_12_yolov1/VOC2007/Annotations/'
Annotations = '/media/iaes/新加卷/wangzy/VOC07_12_yolov1/VOC2012/Annotations/'
xml_files = os.listdir(Annotations)
xml_files.sort()

count = 0
for xml_file in xml_files:
    count += 1
    # if xml_file.split('.')[0] not in lines:
    #     # print(xml_file.split('.')[0])
    #     continue
    image_path = xml_file.split('.')[0] + '.jpg'
    results = parse_rec(Annotations + xml_file)
    if len(results)==0:
        print(xml_file)  # all difficult == 1
        continue
    txt_file.write(image_path)
    # num_obj = len(results)
    # txt_file.write(str(num_obj)+' ')
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = VOC_CLASSES.index(class_name)
        txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
    txt_file.write('\n')
    #if count == 10:
    #    break
txt_file.close()
print(count)