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
from scipy.spatial.distance import pdist

def mDistance(a,b,cov_vec):
    '''
    :param a: 向量a
    :type a: np.array
    :param b: 向量b
    :type b: np.array
    :param cov_vec: 向量a和向量b的协方差矩阵
    :type cov_vec: np.array
    :return -> 向量a和向量b的马氏距离
    :rtype: float
    '''
    x=np.vstack((a,b))
    rev_vec=np.linalg.pinv(cov_vec)    
    tmp=a-b
    res=np.sqrt(np.dot(np.dot(tmp,rev_vec),tmp.T))

    return res

def get_data(data_dir):
    '''
    : divide_image_and_label方法: 读取PCA处理过的240张图片的训练集和160张图片的测试集
    : data_dir: str, PCA处理过的训练集和测试集数据路径
    : return: (train_image, train_label, test_image, test_label) ,train_image: np.array, 训练集; train_label: list, 训练集标签; test_image: np.array, 测试集; test_label: list, 测试集标签
    '''
    label=[0 for i in range(400)]
    for i in range(40):
        for k in range(10):
            label[i*10+k]=i+1  

    train_image=np.load(data_dir+'/train_image_4.npy')
    train_label=[]
    test_image=np.load(data_dir+'/test_image_4.npy')
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
    #print(train_image)
    #print(train_image.shape)
    #print(train_label)
    #print(test_image.shape)
    #print(test_label)

    #2. 计算训练集中各个类别的均值和协方差
    avg_list=[0 for i in range(40)]
    cov_list=[0 for i in range(40)]
    for i in range(40):
        lis=[]   #6行分别代表6个样本，4列分别代表4个特征（变量）
        avg=np.array([0.0 for i in range(4)])
        for k in range(6):
            avg+=train_image[i*6+k]
            lis.append(train_image[i*6+k])
        avg=avg/6.0
        lis=np.array(lis)
        cov=np.cov(lis,rowvar=False)
        avg_list[i]=avg
        cov_list[i]=cov
    #print(avg_list[1])
    #print(cov_list[1])

    #3. 进行分类，计算每个测试集中的数据到各个类的均值向量的马氏距离，取最小的为所属分类标签
    predict_label=[0 for i in range(160)]
    
    for i in range(160):
        dis=[0 for i in range(40)]
        for k in range(40):
            dis[k]=mDistance(test_image[i],avg_list[k],cov_list[k])
        predict_label[i]=dis.index(min(dis))+1    #注意下标要加1才得到标签
    print("使用马式距离的预测结果:")
    print(predict_label)

    
    #3. 计算准确度
    hit=0.0        #命中的样本适量
    total=160.0    #测试集总共的样本数量
    for i in range(len(predict_label)):
        if predict_label[i]==test_label[i]:
            hit+=1
    acc=hit/total
    print("预测准确率 accuracy = ",acc)


    
