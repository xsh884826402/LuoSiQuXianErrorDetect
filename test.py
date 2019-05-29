import sys
import argparse
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import xlrd
from datetime import date,datetime
import matplotlib.pyplot as plt
# import xlsxwriter
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')
from keras.optimizers import Adam
import shutil,os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
from math import sqrt
from scipy.spatial.distance import pdist
from scipy.stats import chisquare
import scipy.stats
from sklearn.cluster import SpectralClustering
import pywt
from pyemd import emd_samples


def DDuni (DDMatrix):
    DDmax    = DDMatrix.max()
    DDshape =  DDMatrix.shape
    DDrows  =  DDshape[0]
    DDcols  =  DDshape[1]
    print (DDrows,'--', DDcols)
    DDreturn = np.zeros ((DDrows, DDcols))
    for rowi in range(DDrows):
        for colj in range(DDcols):
            if DDMatrix[rowi][colj] <= 1e-12:
                   DDreturn[rowi][colj] = 0
            else:
                 DDreturn[rowi][colj] = 200 - (DDMatrix[rowi][colj] / DDmax) * 100
    return DDreturn

def mycopyfile(srcfile,dstfile):#拷贝文件
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件

def get_data():#读取数据，根据cos值构造相似矩阵

    #unidata = pd.read_excel('C:/Users/hxk/Desktop/距离测试/distest.xlsx', header=None, skiprows=None)  # 训练数据文件
    unidata = pd.read_excel('E:/Project/螺丝拧紧/第一批数据/原始数据预处理_excel文件/datauni1_512.xlsx', header=None, skiprows=None)  # 训练数据文件
    datashape =unidata.shape
    data_rows = datashape[0]
    #data_cols = datashape[1]
    test_sub = 1900
    vsize = data_rows-test_sub
    ddpearson = np.zeros((vsize, vsize))
    ddbrc = np.zeros((vsize, vsize))
    ddkl = np.zeros((vsize, vsize))
    ddjs = np.zeros((vsize, vsize))
    ddemdwave = np.zeros((vsize, vsize))
    vend = 128 # 数据长度

    for vi in range(vsize):
        #vectori = abs(unidata.values[vi][0:vend])  # 第 i 个曲线向量
        vectori = (unidata.values[vi][0:vend])  # 第 i 个曲线向量
        # 小波分解处理
        mywavestr = 'sym2'
        mywavlevel = 2
        coeffsvi = pywt.wavedec(vectori, mywavestr, level=mywavlevel)
        waveLdata = coeffsvi[0]
        vecroti_wave    = waveLdata.astype(int)
        # 小波分解处理完成
        for vj in range(vsize):
            #vectorj = abs(unidata.values[vj][0:vend])   # 第 j 个曲线向量
            vectorj = (unidata.values[vj][0:vend])  # 第 j 个曲线向量
            vectorij = np.vstack([vectori, vectorj])  # i,j 向量合并
            # 小波分解处理
            coeffsvj = pywt.wavedec(vectorj, mywavestr, level=mywavlevel)
            waveLdata_vj = coeffsvj[0]
            vecrotj_wave = waveLdata_vj .astype(int)
            # 小波分解处理完成
            # 为 KL 散度距离准备数据
            vectori_sum = sum(vectori)
            vectorj_sum = sum(vectorj)
            vkl_i = vectori / vectori_sum + 1e-10
            vkl_j = vectorj / vectorj_sum + 1e-10     # 为 KL 散度距离准备数据
            # pearson系数
            distpearson = pdist(vectorij, 'cosine')
            ddpearson[vi][vj] = distpearson
            # 布雷柯蒂斯距离
            distbrc = pdist(vectorij, 'braycurtis')
            ddbrc[vi][vj] = distbrc
            # JS 散度
            # vectork = (vectori + vectorj) / 2
            #distjs = 0.5 * scipy.stats.entropy(vectori, vectork) + 0.5 * scipy.stats.entropy(vectorj, vectork)
            #ddjs[vi][vj] = distjs
            # KL 散度
            #distkl = scipy.stats.entropy(vkl_i, vkl_j)
            #ddkl[vi][vj] = distkl
            # EMD 小波处理
            distemdwave= emd_samples(vecroti_wave , vecrotj_wave , bins=50)
            ddemdwave[vi][vj] = distemdwave
            # EMD小波处理
    ddpearsonuni   = DDuni(ddpearson)
    ddbrcuni       = DDuni(ddbrc)
    ddemdwaveuni   = DDuni(ddemdwave)
    #ddjsuni = DDuni(ddjs)
    #ddkluni = DDuni(ddkl)
    ddpearsonuni_rate = 0.33
    ddbrcuni_rate    =  0.33
    ddemdwave_rate    = 0.33
    ddjsuni_rate = 0.25
    ddkluni_rate = 0.25
    SPCM = (     (ddpearsonuni_rate * ddpearsonuni ) + (ddbrcuni_rate  * ddbrcuni)
                 + ( ddemdwave_rate * ddemdwaveuni) )
            # +    (ddjsuni_rate * ddjsuni)            + (ddkluni_rate   * ddkluni)   )

    return SPCM,unidata
def spectral_cluster():#谱聚类

    adj_mat,unidata = get_data()
    cluster_num =2
    #sc = SpectralClustering( cluster_num , affinity='precomputed', n_init=3000, assign_labels='discretize')
    sc = SpectralClustering( cluster_num , affinity='precomputed', n_init=3000,
                            assign_labels='discretize')
    sc.fit(adj_mat)
    # Compare ground-truth and clustering-results
    print('spectral clustering')
    #print(sc.labels_)#输出标签
    print('sc长度',len(sc.labels_))

    class_array = [[100 for i in range(0)] for j in range( cluster_num )]
    class_length = np.zeros(cluster_num)
    for ci in range(cluster_num):
       for scj in range((len(sc.labels_))):
           if sc.labels_[scj] == ci:
               filenumber =scj
               class_array[ci].append( filenumber)
               class_length[ci] =  class_length[ci] + 1
    for ci in range(cluster_num):
       print('类编号 = ', ci,'类个数 =',class_length[ci])
       print ('类序号 = ', class_array[ci])
       print ('-----------------------------------')
    # Calculate some clustering metrics
    for i in range(len(sc.labels_)):
        srcfile = './13D归一化图像/' + str(i + 1) + '.jpeg'
        dstfile = './13D分类/' + str(sc.labels_[i]) + '/' + str(i + 1) + '.jpeg'
        mycopyfile(srcfile, dstfile)
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="使用谱聚类方法对数据及你高兴聚类")
    parser.add_argument('-s','--srcfile',help = "源地址")
    parser.add_argument('-d', '--dstfile', help = "目的地址")
    parser.add_argument('--datafile',help ="需要读取的数据的地址")
    args = parser.parse_args()
    print(args.srcfile,args.dstfile,args.datafile)
    # spectral_cluster()
