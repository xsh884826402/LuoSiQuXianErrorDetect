import sys
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import sklearn
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras import backend as K
import pickle
import xlrd
import os
import numpy as np
from datetime import date,datetime
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')
from keras.optimizers import Adam
import shutil,os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


srcpath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/截取后图像归一化/'     # 所有文件图像
negdespath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/LSTM文件输出异常/'  # 存放异常文件图像
posdespath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/LSTM文件输出正常/'  # 存放正常图像图像

train_data_dir = './data/lstmtraindata1.xlsx'
test_data_dir = './data/datauni_512_928.xlsx'
train_data = pd.read_excel(train_data_dir)
test_data = pd.read_excel(test_data_dir)
train_data_row_num,train_data_col_num =train_data.shape[0],train_data.shape[1]

# lstmtrain_data =  pd.read_excel('C:/Users/hxk/Desktop/pythontest1/lstmtraindata1.xlsx',header=None, skiprows=None)  # 训练数据文件
# datauni_all    =  pd.read_excel('C:/Users/hxk/Desktop/pythontest1/datauni5k.xlsx',header=None, skiprows=None)       #  评价数据文件
# datashape = lstmtrain_data.shape
# lstmtrain_data_rows = datashape[0]
# lstmtrain_data_cols= datashape[1]


traindata_features = 100
lstmdata_all_list  =[]
labeldata=[]
samples_number_train = 58

for i in range(samples_number_train):
    temp_lstm_data = train_data.values[i][0:traindata_features]/1000
    lstmdata_all_list.append(temp_lstm_data.tolist())
    if i < 27:
        labeldata.append([0,1])
    else:
        labeldata.append([1,0])



for lstmtrain_datai  in range (samples_number_train):
    tmp_alldata = lstmtrain_data.values[lstmtrain_datai]
    tmp_lstmdata = tmp_alldata[0:traindata_features]/1000
    #print (tmp_lstmdata)
    #lstmdata_all_list.append(tmp_alldata.tolist())
    #lstmdata_all_list = lstmdata_all_list +  tmp_lstmdata.tolist()
    lstmdata_all_list.append(tmp_lstmdata.tolist())
    if ( lstmtrain_datai < 27):
        labeldata.append([0,1])
        #print ( 'file index  = ',tmp_alldata[100],'datai =',lstmtrain_datai )
    else:
        labeldata.append([1,0])
        #print('file index  = ', tmp_alldata[100], 'datai =', lstmtrain_datai)
#print(labeldata)

labeldata_array  = np.array(labeldata)

#print('Debug: lstmdata_all_list =  ', lstmdata_all_list)
#print('Debug: lstmdata_all_list len  =  ', len(lstmdata_all_list))

lstmdata_all_array= np.array(lstmdata_all_list)

#print('Debug: lstmdata_all_array   =  '         , lstmdata_all_array)
#print('Debug: lstmdata_all_array shape   =  '  ,  lstmdata_all_array.shape)


def train_test(labeldata_array ):
    global samples
    global losses

    for i in range(1):
        K.clear_session()
        losses = []
        timesteps = 1
        dim = 100

        def handleLoss(loss):
            global losses
            losses += [loss]
            print(loss)

        class LossHistory(Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
               # print(logs.get('loss'))
                handleLoss(logs.get('loss'))

        Nnum = 50
        model = Sequential()
       # model.add(Dense(100))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True, dropout_W=0.2, dropout_U=0.2))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True, dropout_W=0.2, dropout_U=0.2))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True, dropout_W=0.2, dropout_U=0.2))
        model.add(LSTM(Nnum, input_shape=(timesteps, dim), return_sequences=True))
        model.add(Dense(2))
        #model.add(Dense(50, activation='sigmoid'))
        #model.add(Dense(2,activation='softmax'))
        #model.add(Dense(1,activation='softmax' ))# 数据维度相关
       #model.compile(loss='mae', optimizer='adam')
        adam = Adam(lr=0.001)
        model.compile(loss='mae', optimizer=adam)
        #r2_score
        def train(data,datay,samples):

            #data.shape = (52,100,1)
            #datay.shape= (52,1,1)
            model.fit(data, datay, epochs=250, batch_size=16, verbose=0, shuffle=False,
                      callbacks=[LossHistory()], validation_data=(data, datay),)


        # 使用同一个样本和所有待测试样本比较
        for i in range(5):
            print("---------------- 循环次数 = ",i,'------------------------------')
            tmp_lstm = lstmdata_all_array
            tmp_lstm = tmp_lstm.reshape(samples_number_train,1,100)
            labeldata_array = labeldata_array.reshape(samples_number_train,1,2)
            train(tmp_lstm,labeldata_array,samples_number_train*100)
        #print(tmp_lstm)
        #core, acc = model.evaluate(tmp_lstm, labeldata_array, batch_size=52)
        #print ( score  ,acc)
        for tti in range(5000):
           test1 = datauni_all.values[tti]
           test11 =test1[0:traindata_features] / 1000
           test111 = test11.reshape(1,1,100)
           #print(test111)
           prediction = model.predict(test111)
           index = np.argmax(prediction)
           #print('数字 = ', tti, '类别=', index)
           if (index == 1):
              str_i = tti+1
              filenum_str = str(str_i)
              desfilename = negdespath + filenum_str + '.jpeg'
              srcfilename = srcpath + filenum_str + '.jpeg'
              shutil.copyfile(srcfilename, desfilename)
              print('文件编号 = ',tti ,'异常类别=', index, '预测矩阵=  ',prediction)
           else:
               str_i = tti + 1
               filenum_str = str(str_i)
               desfilename = posdespath + filenum_str + '.jpeg'
               srcfilename = srcpath + filenum_str + '.jpeg'
               shutil.copyfile(srcfilename, desfilename)
               print('文件编号 = ', tti, '正常类别=', index, '预测矩阵=  ', prediction)
             # print(test111)



if __name__ == '__main__':
    train_test(labeldata_array )


