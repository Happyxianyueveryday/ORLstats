#coding with utf-8

import numpy as np
import struct
import random
import os
import sys
import matplotlib.pyplot as plt
from functools import reduce
from PIL import Image
from sklearn.decomposition import PCA

def eDistance(a,b):
    '''
    :param a: 向量a
    :type a: np.array
    :param b: 向量b
    :type b: np.array
    :return -> 向量a和向量b的欧式距离
    :rtype: float
    '''
    res=np.sqrt(np.sum(np.square(a-b)))
    return res

def get_data(data_dir):
    '''
    : get_data方法: 读取PCA处理过的240张图片的训练集和160张图片的测试集
    : data_dir: str, PCA处理过的训练集和测试集数据路径
    : return: (train_image, train_label, test_image, test_label) ,train_image: np.array, 训练集; train_label: list, 训练集标签; test_image: np.array, 测试集; test_label: list, 测试集标签
    '''
    label=[0 for i in range(400)]
    for i in range(40):
        for k in range(10):
            label[i*10+k]=i+1  

    train_image=np.load(data_dir+'/train_image.npy')
    train_label=[]
    test_image=np.load(data_dir+'/test_image.npy')
    test_label=[]
    
    for i in range(40):
        for k in range(6):
            train_label.append(label[i*10+k])
    
    for i in range(40):
        for k in range(6,10):
            test_label.append(label[i*10+k])
        
    return (train_image,train_label,test_image,test_label)


if __name__ == "__main__":
    
    #1. 从PCA处理的结果(.npy)文件中读取训练集和测试集的数据，并生成训练集和测试集的标签列表
    train_image, train_label, test_image, test_label=get_data(sys.path[0])
    #print(train_image.shape)
    #print(train_label)
    #print(test_image.shape)
    #print(test_label)


    #2. 对于测试集中的160个向量（代表一个图片）中的每一个，分别计算该向量到训练集的240个向量之间的欧式距离，将与之距离最小的那个训练集向量的标签视为预测的标签记录
    predict_label=[0 for i in range(160)]
    
    for i in range(160):
        dis=[0 for i in range(240)]
        for k in range(240):
            dis[k]=eDistance(test_image[i],train_image[k])
        predict_label[i]=train_label[dis.index(min(dis))]
    print("使用欧式距离的预测结果:")
    print(predict_label)
    
    #3. 计算准确度
    hit=0.0        #命中的样本适量
    total=160.0    #测试集总共的样本数量
    for i in range(len(predict_label)):
        if predict_label[i]==test_label[i]:
            hit+=1
    acc=hit/total
    print("预测准确率 accuracy = ",acc)
            



    


    
