#coding with utf-8

import numpy as np
import struct
import random
import os
import sys
import matplotlib.pyplot as plt
from functools import reduce
from PIL import Image

# 任务1: 对PCA处理后的训练集和测试集进行聚类并计算准确率

samples_size=400     #总的聚类图像数量400

def get_image(train_data_dir,test_data_dir):
    '''
    : get_image方法: 从PCA处理后的数据中读取训练集和测试集的向量
    : train_data_dir: str, 训练集PCA处理结果(.npy文件)的绝对路径
    : test_data_dir: str, 测试集PCA处理结果(.npy文件)的绝对路径
    : return: np.array， PCA处理结果的训练集和测试集的合并数据，大小为400*40
    '''
    train_image=np.load(train_data_dir)
    test_image=np.load(test_data_dir)
    res=np.concatenate((train_image,test_image))

    return res

def get_label(): 
    '''
    : get_label方法: 该方法读取经过PCA处理后的训练集和测试集的向量（图像）标签，先是6个为一类，总计240个的训练集向量标签，后是4个为一类，总计160的测试集向量标签，将测试集和训练集向量标签列表合并即可得到总体标签
    : return: list, PCA处理后的图像的标签
    '''
    label=[0 for i in range(400)]
    for i in range(40):
        for k in range(10):
            label[i*10+k]=i   #附注：将类别1,...,40替换为0,...,39，方便后续的统计步骤

    train_label=[]
    test_label=[]
    for i in range(40):
        for k in range(6):
            train_label.append(label[i*10+k])
    
    for i in range(40):
        for k in range(6,10):
            test_label.append(label[i*10+k])
    
    res=train_label+test_label

    return res



if __name__ == "__main__":
    #1. 读取图像数据和图像标签

    im=get_image(sys.path[0]+'/train_image.npy',sys.path[0]+'/test_image.npy')
    label = get_label()
    #print(len(im))
    #print(im[0].shape)
    #print(label)
    #print(len(label))


    #2. 将原始图像向量化
    #经过PCA处理的图像已经经过向量化，故本步骤可以直接跳过


    #3. 对向量化的图像进行Kmeans聚类
    #这时im转化为如下的情况，即二维矩阵im总共含有samples_size=400个向量/行，每一个行包括一个40维的向量，一个40维的向量就是一个样本

    #3.1 首先设置初始质心，我们在初始状态下直接从样本集中随机选定40个样本作为质心
    focus=[]
    for i in range(40):
        focus.append(random.choice(im))
    focus=np.array(focus)

    #3.2 然后进行迭代计算处理，每一轮迭代中首先先将samples_size个样本根据到质心欧式距离中的最短距离，归类到对应的聚类中，然后重新计算每一聚类中的样本点的平均值作为新的质心，新的质心代替原有的质心
    res=[]              
    role=0
    print("Iteration begin!")
    while 1:
        print("rounds = ",role)
        role=role+1
        res=[[] for i in range(40)]        #手写体数字分为40类，聚类的数量为40
        for j in range(samples_size):
            dis=[0 for i in range(40)]     #样本点im[j]和40个质心的欧式距离
            for k in range(40):
                dis[k]=np.linalg.norm(im[j]-focus[k])  
            index=dis.index(min(dis))        #欧式距离中的最小值对应的下标就是所属的聚类的编号
            res[index].append(j)           #将样本点im[j]的编号j放入第index个聚类中
        #当一轮迭代完成后，重新计算各个聚类的质心
        avg=np.array([[0.0 for i in range(40)] for k in range(40)])
        for m in range(40):
            if len(res[m])!=0:   #聚类中有样本才重新计算质心，否则直接取原质心
                for n in range(len(res[m])):
                    avg[m]=avg[m]+im[res[m][n]]
                avg[m]=avg[m]/len(res[m])
            else:
                avg[m]=focus[m]
        #print("avg=",avg)
        #print("focus=",focus)
        if (avg==focus).all():
            break
        else:
            focus=avg
    print("Iteration end!")
    print("final result:")
    print(res)


    #4. 聚类步骤完成，开始计算精度
    print("calculating accuracy...")

    #4.1 首先计算结果中各个聚类中出现次数最多的标签
    tags=[0 for i in range(40)]       #40个聚类的标签
    for i in range(40):
        counts=[0 for i in range(40)]
        for k in range(len(res[i])):
            counts[label[res[i][k]]]+=1
        tags[i]=counts.index(max(counts))  #聚类中出现次数最多的数字的标签作为这个聚类的标签
    
    #4.2 然后计算准确度
    correct_sz=0       #准确预测的样本数量
    for i in range(40):
        for k in range(len(res[i])):
            if label[res[i][k]]==tags[i]:
                correct_sz+=1
    
    accuracy=float(correct_sz/samples_size)
    print("accuracy = ",accuracy)



            


            


        



