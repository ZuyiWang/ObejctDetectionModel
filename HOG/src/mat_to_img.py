'''
@author: Wangzy
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zuyiwang@zju.edu.cn
@software: 
@file: mat_to_img.py
@time: 2020/11/13 15:12
@desc:
'''
from PIL import Image
import numpy as np


def readData():
    image_dir = r"G:\Object Detection Model\HOG\data\1.jpg"

    x = Image.open(image_dir)  # 打开图片
    data = np.asarray(x)  # 转换为矩阵
    print(data.shape)  # 输出矩阵

    image = Image.fromarray(data)  # 将之前的矩阵转换为图片
    image.show()  # 调用本地软件显示图片，win10是叫照片的工具


readData()