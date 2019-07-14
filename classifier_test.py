

@app.route('/classfier_test',methods={'post','get'})
def classifier_test():
    model_dir = request.values.get('model_dir').strip()
    test_data_dir = request.values.get('test_data_dir').strip()
    test_data = pd.read_excel(test_data_dir)

    model = models.load_model(model_dir)
    for tti in range(5999):
        test = test_data.values[tti]
        test = test[0:traindata_features] / uni_max_value
        test = test.reshape(1, 1, 100)
        # print(test111)
        prediction = model.predict(test)
        index = np.argmax(prediction)
        # print('数字 = ', tti, '类别=', index)
        if (index == 1):
            str_i = tti + 1
            filenum_str = str(str_i)
            desfilename = negdespath + filenum_str + '.jpeg'
            srcfilename = srcpath + filenum_str + '.jpeg'
            shutil.copyfile(srcfilename, desfilename)
            print('文件编号 = ', tti, '异常类别=', index, '预测矩阵=  ', prediction)
        else:
            str_i = tti + 1
            filenum_str = str(str_i)
            desfilename = posdespath + filenum_str + '.jpeg'
            srcfilename = srcpath + filenum_str + '.jpeg'
            shutil.copyfile(srcfilename, desfilename)
            print('文件编号 = ', tti, '正常类别=', index, '预测矩阵=  ', prediction)




if __name__ == '__main__':
    train_test(labeldata_array,train = False,model_dir  = './model/model.h5')


