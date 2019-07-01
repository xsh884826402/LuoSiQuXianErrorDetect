import numpy as np
import pandas as pd

#unidata = pd.read_excel('D:/距离测试/12dydataall10000.xlsx', header=None, skiprows=None)  # 训练数据文件
y_data = pd.read_excel('D:/距离测试/12dydatatest.xlsx', header=None, skiprows=None)  # 训练数据文件

datashape = y_data.shape
data_rows = datashape[0]
data_cols = datashape[1]
data_len = np.zeros(data_rows)
#for  di  in range(10000):
    #data_len[di] = unidata.values[di][0]
data_len = y_data.values[:,0]#取二维数组每一行的第一个元素
print(data_len)

data_len_max = int(np.max(data_len))
print ( data_rows, data_cols)
print('数据最大长度=' ,data_len_max )

train_data = np.zeros((int(data_rows),int(data_len_max )))
train_data_row_max  = np.zeros(int(data_rows))
errordata = np.ones(data_len_max)
for di in range(data_rows):
    train_data[di] = y_data.values[di][1:data_len_max+1]
    train_data_row_max[di] = np.max (train_data[di])
    if  (train_data_row_max [di] == 0 ):
        train_data[di]= errordata/1000000
    else:
        train_data[di] = train_data[di]/train_data_row_max[di]
        train_data[di] = train_data[di]
#print( 'train data shape = ', train_data.shape)


np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input,LSTM, TimeDistributed,Reshape
import matplotlib.pyplot as plt

# 数据预处理
x_train = train_data.astype('float32')  # minmax_normalized
x_test =  train_data.astype('float32') # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 压缩特征维度至2维
encoding_dim = 128
codesize = 1024
# this is our input placeholder
input_img = Input(shape=(data_len_max,))
act_fun_str = 'relu'
act_fun_str1 = 'tanh'
#act_fun_str1 = 'tanh'
timesteps = 1
dim = codesize
# 编码层
#encoded =  TimeDistributed(Dense(int(codesize),   activation= act_fun_str))(input_img)
encoded =  Dense(int(codesize),   activation= act_fun_str)(input_img)
print(encoded)
#Reshape((trainY.shape[1], trainY.shape[2])))
encoded = Reshape((-1,codesize))(encoded)
encoded = LSTM(int(codesize/2),activation= act_fun_str,return_sequences=False)(encoded)
encoded = Reshape((-1,int(codesize/2)))(encoded)
encoded = LSTM(int(codesize/4),activation= act_fun_str,return_sequences=False)(encoded)
encoded = Reshape((-1,int(codesize/4)))(encoded)
encoder_output = LSTM(int(codesize/8),activation= act_fun_str,return_sequences=False)(encoded)
#encoded = Dense(int(codesize/4), activation= act_fun_str)(encoded)
#encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(int(codesize/4), activation= act_fun_str)(encoder_output)
decoded = Dense(int(codesize/2),activation= act_fun_str)(decoded)
decoded = Dense(int(codesize/1), activation= act_fun_str)(decoded)
decoded = Dense(data_len_max, activation= act_fun_str1)(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train, epochs=1000, batch_size=500, shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
ende_imgs = autoencoder.predict(x_test)



encodefile='D:/距离测试/12sencoderesult.xlsx'
endefile='D:/距离测试/12dendecoderesult.xlsx'
enfile_data = pd.DataFrame(encoded_imgs)
endefile_data = pd.DataFrame(ende_imgs)

enfile_data.to_excel(encodefile, sheet_name='encode Data', index=None, header=None)
endefile_data.to_excel(endefile, sheet_name='encode Data', index=None, header=None)

print ('encode shape = ',encoded_imgs.shape)
print ('encode decode shape = ',ende_imgs.shape)

plt.plot(ende_imgs[0])
print(ende_imgs[0])
plt.title('encode  line:0  ')
plt.show()
plt.plot(x_test[0])
plt.title('original  line :0 ')
plt.show()
plt.plot(ende_imgs[3])
plt.title('encode  line:3 ')
plt.show()
plt.plot(x_test[3])
plt.title('original  line :3 ')
plt.show()

plt.plot(ende_imgs[1])
plt.title('encode  line 1 ')
plt.show()
plt.plot(x_test[1])
plt.title( 'original  line 1')
plt.show()
plt.plot(ende_imgs[2])
plt.title('encode  line 2 ')
plt.show()
plt.plot(x_test[2])
plt.title( 'original  line 2')
plt.show()


plt.plot(encoded_imgs[0])
plt.title( 'original  line 0')
plt.show()
plt.plot(encoded_imgs[3])
plt.title( 'original  line 3')
plt.show()
plt.plot(encoded_imgs[1])
plt.title( 'original  line 1')
plt.show()
plt.plot(encoded_imgs[2])
plt.title( 'original  line 2')
plt.show()

#plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=x_test, s=3)
#plt.colorbar()
#plt.show()


