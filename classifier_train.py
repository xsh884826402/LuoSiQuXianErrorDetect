
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


# srcpath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/截取后图像归一化/'     # 所有文件图像
# negdespath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/LSTM文件输出异常/'  # 存放异常文件图像
# posdespath = 'C:/Users/hxk/Desktop/螺丝拧紧曲线matlab/分类代码20180802/LSTM文件输出正常/'  # 存放正常图像图像
def get_data(train_data_dir):
    srcpath = 'E:/Project/螺丝拧紧/第一批数据/可视化图片/截取后图像归一化/'     # 所有文件图像
    negdespath = 'E:/Project/螺丝拧紧/第一批数据/可视化图片/LSTM文件输出异常/'  # 存放异常文件图像
    posdespath = 'E:/Project/螺丝拧紧/第一批数据/可视化图片/LSTM文件输出正常/'  # 存放正常图像图像
    # shape = 40 * 256
    train_data_dir = './data/lstmtraindata1.xlsx'
    # shape = 6000 * 512
    test_data_dir = './data/datauni1_512_928.xlsx'
    train_data = pd.read_excel(train_data_dir)
    train_data_row_num,train_data_col_num =train_data.shape[0],train_data.shape[1]

    traindata_features = 100
    lstmdata_all_list  =[]
    labeldata=[]
    samples_number_train = 39
    uni_max_value = 1000
    for i in range(samples_number_train):
        temp_lstm_data = train_data.values[i][0:traindata_features]/uni_max_value
        lstmdata_all_list.append(temp_lstm_data.tolist())
        #猜测 这里的27可能指的是做demo的时候前27条数据的是正类
        if i < 27:
            labeldata.append([0,1])
        else:
            labeldata.append([1,0])
    labeldata_array = np.array(labeldata)
    lstmdata_all_array = np.array(lstmdata_all_list)
    return lstmdata_all_array,labeldata_array

@app.route('/classfier_train',methods={'post','get'})
def classifier_train():
    model_dir = request.values.get('model_dir').strip()
    train_data_dir = request.values.get('train_data_dir').strip()
    #输入数据
    train_data,label_data = get_data(train_data_dir)
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
            tmp_lstm = train_data
            tmp_lstm = tmp_lstm.reshape(samples_number_train,1,100)
            labeldata_array = label_data.reshape(samples_number_train,1,2)
            train(tmp_lstm,label_data,samples_number_train*100)
            model.save(model_dir)

        #print(tmp_lstm)
        #core, acc = model.evaluate(tmp_lstm, labeldata_array, batch_size=52)
        #print ( score  ,acc)



if __name__ == '__main__':
    train_test(labeldata_array,train = False,model_dir  = './model/model.h5')


