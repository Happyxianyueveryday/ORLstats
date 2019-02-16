#coding with utf-8

import numpy as np
import struct
import random
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing

# 任务1: 对原始图像划分训练集和测试集，并进行PCA处理

samples_size=400     #总的聚类图像数量400

def get_image(file_dir):
    '''
    : get_image方法: 从数据集中按顺序读取400张图像
    : file_dir: str, 图像根文件夹的绝对路径
    : return: list, 原始图像列表，按顺序排列，每10个元素为一类，类名也按照顺序排序，每个元素的形式为112*92的np.array
    '''

    res=[]
    sub_dir_list=os.listdir(file_dir)   
    sub_dir_list.sort(key=lambda x:int(x[1:]))   #对路径进行排序
    for s in sub_dir_list:
        sub_dir=file_dir+'/'+s
        sub_sub_dir_list=os.listdir(sub_dir)     #对路径进行排序
        sub_sub_dir_list.sort(key=lambda x:int(x[:-4]))
        for k in sub_sub_dir_list:
            image_dir=sub_dir+'/'+k
            im=Image.open(image_dir)
            im=np.array(im) 
            res.append(im)
            #print(image_dir)
    return res
 
def get_label(): 
    '''
    : get_label方法: 从数据集中按顺序读取400张图像的标签
    : return: list, 原始图像标签
    '''

    #由于get_image按顺序读取，因此每10个为一类，直接生成标签即可
    res=[0 for i in range(400)]
    for i in range(40):
        for k in range(10):
            res[i*10+k]=i+1     #类别标签为1,...,40
    return res    

def divide_image_and_label(image, label):
    '''
    : divide_image_and_label方法: 将原始数据集划分为240张图片的训练集和160张图片的测试集
    : image: list, 原始图像列表，按顺序排列，每10个元素为一类，类名也按照顺序排序，每个元素的形式为112*92的np.array
    : label: list, 原始图像标签
    : return: (train_image, train_label, test_image, test_label) ,train_image: 训练集, train_label: 训练集标签, test_image: 测试集, test_label: 测试集标签，均为list
    '''
    train_image=[]
    train_label=[]
    test_image=[]
    test_label=[]
    
    for i in range(40):
        for k in range(6):
            train_image.append(image[i*10+k])
            train_label.append(label[i*10+k])
    
    for i in range(40):
        for k in range(6,10):
            test_image.append(image[i*10+k])
            test_label.append(label[i*10+k])
        
    return (train_image,train_label,test_image,test_label)

def flatten_image(im):
    '''
    : flatten_image方法: 将图像集合向量化，每个图像大小为112*92
    : return: np.array, 向量化后的图片
    '''
    res=[0 for i in range(len(im))]
    for i in range(len(im)):
        temp=[]
        for k in range(112):
            for m in range(92):
                temp.append(im[i][k][m])
        res[i]=temp
    
    res=np.array(res)   #将im转化为np.array，label此时仍然为list
    return res

def pca(X,k):
    '''
    : X: 原始数据集，每一列为一个特征，每一行为一个样本
    : k: 主成分分析(PCA)需要保留的特征数
    '''

    n_samples, n_features = X.shape
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
    #1. 数据标准化
    norm_X=X-mean
    #2. 计算散布矩阵/协方差矩阵
    scatter_matrix=np.cov(norm_X,rowvar=False)
    #3. 计算特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    #4. 将特征值和对应的特征向量从大到小排列
    eig_pairs.sort(reverse=True)
    #5. 选择最大的k个特征值对应的特征向量，变换得到PCA降维结果矩阵
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    data=np.dot(norm_X,np.transpose(feature))
    return data
 
if __name__ == "__main__":
    #1. 读取图像数据和图像标签

    image=get_image(sys.path[0]+'/att_faces')
    label = get_label()
    #print(len(im))
    #print(im[0].shape)
    #print(label)
    #print(len(label))

    #2. 分割训练集和测试集
    train_image,train_label,test_image,test_label=divide_image_and_label(image,label)
    #print(len(train_image))
    #print(len(train_label))
    #print(len(test_image))
    #print(len(test_label))
    #print(train_image[0].shape)
    #print(test_image[0].shape)
    #print(train_label)
    #print(test_label)

    #3. 将原始图像向量化
    #这时的train_image,test_image中的每一个元素为一个112*92的灰度矩阵，对应于一张灰度图
    #这时的train_label,test_label中的每一个元素为一个十进制数，对应于一张手写体识别灰度图的标签
    train_image=flatten_image(train_image)
    test_image=flatten_image(test_image)
    #print(train_image.shape)
    #print(test_image.shape)

    #4. 将训练集和测试集先进行进行主成分分析(PCA)，分别由240*10304，160*10304压缩为240*40，160*40

    train_image_1=pca(train_image,40)
    test_data_1=pca(test_data,40)
    train_image_2=pca(train_image,4)
    test_data_2=pca(test_data,4)

    print("PCA步骤完成")



    


