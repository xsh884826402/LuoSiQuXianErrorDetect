
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import shutil,os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
from math import sqrt
from numpy import random,mat
from scipy.spatial.distance import pdist
from scipy.stats import chisquare
import scipy.stats
from sklearn.cluster import SpectralClustering
import pywt
from pyemd import emd_samples
from multiprocessing import Pool
import random
# unidata = pd.read_excel('D:/项目_MFK/螺丝曲线检测/第三批数据/12sencoderesult.xlsx', header=None, skiprows=None)  # 训练数据文件
unidata = pd.read_excel('./datauni1_512.xlsx', header=None, skiprows=None)
datashape = unidata.shape
#print('datashape',datashape)
data_rows = datashape[0]
# data_cols = datashape[1]
test_sub = 500
vsize = data_rows - test_sub
ddpearson = np.zeros((vsize, vsize))
ddbrc = np.zeros((vsize, vsize))
ddkl = np.zeros((vsize, vsize))
ddjs = np.zeros((vsize, vsize))
ddemdwave = np.zeros((vsize, vsize))
vend = 128  # 数据长度

def DDuni (DDMatrix):
    DDmax    = DDMatrix.max()
    DDshape =  DDMatrix.shape
    DDrows  =  DDshape[0]
    DDcols  =  DDshape[1]
    #print (DDrows,'--', DDcols)
    DDreturn = np.zeros ((DDrows, DDcols))
    for rowi in range(DDrows):
        for colj in range(DDcols):
            if DDMatrix[rowi][colj] <= 1e-12:
                   DDreturn[rowi][colj] = 0
            else:
                 DDreturn[rowi][colj] = 200 - (DDMatrix[rowi][colj] / DDmax) * 100
    return DDreturn
#传入参数池list
#返回参数：dict 第一个元素是行索引，第二个是行向量
def paral_process_similirity(arg):

    index1 = arg
    print('index1',index1)
    # randomNum = random.randint(2,10)
    # time.sleep(randomNum)
    vectori = (unidata.values[index1][0:vend])  # 第 i 个曲线向量
    #print('vectori:',vectori)
    # 小波分解处理
    mywavestr = 'sym2'
    mywavlevel = 2
    coeffsvi = pywt.wavedec(vectori, mywavestr, level=mywavlevel)
    waveLdata = coeffsvi[0]
    vecroti_wave = waveLdata.astype(int)

    list_ddpearson = []
    list_ddbrc = []
    list_ddemdwave = []
    for vj in range(vsize):
        # index2=arg
        # print("index1: ",index1, "index2: ", index2)
        vectorj = (unidata.values[vj][0:vend])  # 第 j 个曲线向量
        # vectorj = (unidata.values[index2][0:vend])
        vectorij = np.vstack([vectori, vectorj])  # i,j 向量合并
        # 小波分解处理
        coeffsvj = pywt.wavedec(vectorj, mywavestr, level=mywavlevel)
        waveLdata_vj = coeffsvj[0]
        vecrotj_wave = waveLdata_vj.astype(int)
        # 小波分解处理完成

        # pearson系数
        distpearson = pdist(vectorij, 'cosine')
        list_ddpearson.append(distpearson[0])

        # 布雷柯蒂斯距离
        distbrc = pdist(vectorij, 'braycurtis')
        list_ddbrc.append(distbrc[0])

        #模态经验分解
        distemdwave = emd_samples(vecroti_wave, vecrotj_wave, bins='auto')
        #print('distemdwave',distemdwave)
        list_ddemdwave.append(distemdwave)
    #print('list_ddemdwave:',list_ddemdwave)
    return list_ddpearson,list_ddbrc,list_ddemdwave
def main():#读取数据，根据cos值构造相似矩阵
    start_time = time.time()
    works = 8
    pool = Pool()

    arglist = []
    for i in range(vsize):
        arglist.append(i)
    list_all= pool.map(paral_process_similirity,arglist)
    pool.close()
    pool.join()
    print('success')
    array_all = np.array(list_all)
    print('list_ddpearson shape',array_all.shape)
    #print('list_ddpearson', array_all)
    print('----------')
    matrix_ddpearson = array_all[:,0,:]
    matrix_ddbrc = array_all[:,1,:]
    matrix_ddemdwave = array_all[:,2,:]
    print('matrix_ddpearson',matrix_ddpearson.shape)
    print('matrix_ddbrc',matrix_ddbrc.shape)
    print('matrix_ddemdwave', matrix_ddemdwave.shape)

    ddpearsonuni   = DDuni(matrix_ddpearson)
    ddbrcuni       = DDuni(matrix_ddbrc)
    ddemdwaveuni   = DDuni(matrix_ddemdwave)
    ddpearsonuni_rate = 0.33
    ddbrcuni_rate    =  0.33
    ddemdwave_rate    = 0.33
    SPCM = (     (ddpearsonuni_rate * ddpearsonuni ) + (ddbrcuni_rate  * ddbrcuni)
                 + ( ddemdwave_rate * ddemdwaveuni) )
    # f1 = open('聚类优化demo.txt', 'w', encoding='utf-8')  # Minweight
    # for i in range(20):
    #     for j in range(20):
    #         #print(matrix_ddpearson[i, j])
    #         a = round(SPCM[i, j], 2)
    #         f1.write(str(a))
    #         f1.write(' ')
    #     f1.write('\n')
    # f1.close()
    end_time = time.time()
    print('time:',end_time-start_time)
if __name__ == '__main__':
    main()