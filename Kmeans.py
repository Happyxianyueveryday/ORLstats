#coding with utf-8

import numpy as np
import struct
import random
import os
import sys
import matplotlib.pyplot as plt
from functools import reduce
from PIL import Image

# 任务1: 对原始图像进行聚类并计算准确率

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
        sub_dir=file_dir+'\\'+s
        sub_sub_dir_list=os.listdir(sub_dir)     #对路径进行排序
        sub_sub_dir_list.sort(key=lambda x:int(x[:-4]))
        for k in sub_sub_dir_list:
            image_dir=sub_dir+'\\'+k
            im=Image.open(image_dir)
            im=np.array(im) 
            res.append(im)
            #print(image_dir)
    return res
 
#从数据集中读取samples_size个手写体图像对应的标签
def get_label(): 
    '''
    : get_label方法: 从数据集中按顺序读取400张图像的标签
    : return: list, 原始图像标签
    '''

    #由于get_image按顺序读取，因此每10个为一类，直接生成标签即可
    res=[0 for i in range(400)]
    for i in range(40):
        for k in range(10):
            res[i*10+k]=i     #附注：将类别1,...,40替换为0,...,39，方便后续的统计步骤
    return res    


 
if __name__ == "__main__":
    #1. 读取图像数据和图像标签

    im=get_image(sys.path[0]+'/att_faces')
    label = get_label()
    #print(len(im))
    #print(im[0].shape)
    #print(label)
    #print(len(label))


    #2. 将原始图像向量化
    #这时的im中的每一个元素为一个112*92的灰度矩阵，对应于一张灰度图
    #这时的label中的每一个元素为一个十进制数，对应于一张手写体识别灰度图的标签

    for i in range(samples_size):
        temp=[]
        for k in range(112):
            for m in range(92):
                temp.append(im[i][k][m])
        im[i]=temp
    
    im=np.array(im)   #将im转化为np.array，label此时仍然为list
    #print(im.shape)
    #print(im[0].shape)


    #3. 对向量化的图像进行Kmeans聚类
    #这时im转化为如下的情况，即二维矩阵im总共含有samples_size=400个向量/行，每一个行包括一个112*92=10304维的向量，一个112*92=10304维的向量就是一个样本

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
        avg=np.array([[0.0 for i in range(10304)] for k in range(40)])
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


            


            


        



