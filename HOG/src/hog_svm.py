'''
@author: Wangzy
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zuyiwang@zju.edu.cn
@software: 
@file: hog_svm.py
@time: 2020/11/12 10:15
@desc: HOG Descriptor
'''
import os
import cv2
import math
import time
import numpy as np
import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from PIL import Image


class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self):
        TrainData = []
        TestData = []
        for childDir in os.listdir(self.filePath):
            if 'data_batch' in childDir:
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                # train = np.reshape(data[str.encode('data')], (10000, 3, 32 * 32))
                # If your python version do not support to use this way to transport str to bytes.
                # Think another way and you can.
                train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (10000, 1))
                fileNames = np.reshape(data[b'filenames'], (10000, 1))
                datalebels = zip(train, labels, fileNames)
                TrainData.extend(datalebels)
            if childDir == "test_batch":
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (10000, 1))
                fileNames = np.reshape(data[b'filenames'], (10000, 1))
                TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData

    def show_img(self):
        '''
        desc: Image.fromarray((x_size, y_size, 3), mode)
        '''
        for childDir in os.listdir(self.filePath):
            if '_batch' in childDir:
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
                img_mat = np.reshape(train[0].T, (32, 32, 3))
                img = Image.fromarray(img_mat)
                img.show()
                cv2.waitKey(0)

    def get_hog_feat(self, image, stride=8, bins=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), sign_grad=True):
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
        sx, sy = image.shape
        # n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        # n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        # n_blocksx = (n_cellsx - bx) + 1   # under the condition: stride == cx
        # n_blocksy = (n_cellsy - by) + 1
        n_blocksx = int(np.floor((sx-cx)//stride))  # number of blocks in x
        n_blocksy = int(np.floor((sy-cy)//stride))  # number of blocks in y
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx-1):
            for j in range(1, sy-1):
                gx[i, j] = image[i, j-1] - image[i, j+1]
                gy[i, j] = image[i+1, j] - image[i-1, j]
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
                if sign_grad:
                    # sign direction [0, 2pi]

                    # when gy>0 & gx<0 and gy<0 & gx<0, theta+180
                    if gx[i, j] <= 0:
                        grad[i, j, 0] += 180
                    # when gy<0 & gx>0, theta+360
                    grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                else:
                    # unsign direction [0, pi]
                    if grad[i, j, 0] < 0:
                        grad[i, j, 0] += 180
        normalized_blocks = np.zeros((n_blocksx, n_blocksy, bx * by * bins))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = grad[x*stride:x*stride+bx*cx, y*stride:y*stride+by*cy]
                hist_block = np.zeros(bx*by*bins, dtype=np.float32)
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[m*cx:(m+1)*cx, k*cy:(k+1)*cy]
                        hist_cell = np.zeros(bins, dtype=np.float32)
                        for i in range(cx):
                            for j in range(cy):
                                if sign_grad:
                                    interval = 360/bins
                                else:
                                    interval = 180/bins
                                n = int(cell[i, j, 0] / interval)
                                # interpolate bilinearly, considering n = 8 respectively
                                if n == bins-1:
                                    hist_cell[n] += ((n+1)*interval - cell[i, j, 0]) / interval * cell[i, j, 1]
                                    hist_cell[0] += (cell[i, j, 0] - n*interval) / interval * cell[i, j, 1]
                                else:
                                    hist_cell[n] += ((n+1)*interval - cell[i, j, 0]) / interval * cell[i, j, 1]
                                    hist_cell[n+1] += (cell[i, j, 0] - n*interval) / interval * cell[i, j, 1]
                                # hist_cell[n] += cell[i, j, 1]
                        hist_block[(k * bx + m) * bins:(k * bx + m + 1) * bins] = hist_cell[:]
                # L2 Norm
                normalized_blocks[x, y, :] = hist_block / np.sqrt((hist_block ** 2).sum() + eps)
        return normalized_blocks.ravel()  # change to 1-D array

    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            # if RGB 3channels, choose the one with the largest magnitude as gradient
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            fd = self.get_hog_feat(gray) #自己写的hog提取函数，下面为skimage提供的hog函数，速度快很多
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        np.save("test_feat.npy", test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = self.get_hog_feat(gray)
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        np.save("train_feat.npy", train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat):
        t0 = time.time()
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(train_feat[:, :-1], train_feat[:, -1])
        predict_result = clf.predict(test_feat[:, :-1])
        num = 0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i, -1]):
                num += 1
        rate = float(num) / len(predict_result)
        t1 = time.time()
        print('The classification accuracy is %f' % rate)
        print('The cast of time is :%f' % (t1 - t0))

    def run(self):
        if os.path.exists("train_feat.npy") and os.path.exists("test_feat.npy"):
            train_feat = np.load("train_feat.npy")
            test_feat = np.load("test_feat.npy")
        else:
            TrainData, TestData = self.get_data()
            train_feat, test_feat = self.get_feat(TrainData, TestData)
        self.classification(train_feat, test_feat)


if __name__ == '__main__':
    filePath = r'..\data\cifar-10-batches-py'
    cf = Classifier(filePath)
    cf.run()