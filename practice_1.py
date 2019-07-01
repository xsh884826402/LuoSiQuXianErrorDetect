input_1 ='abcab'
input_2 = 'cbaab'
def f(input_1,input_2)->bool:
    for i in range(len(input_2)+1):
        if input_2[0:i] in input_1 and input_2[i:] in input_1:
            return True
    return False
flag = f(input_1,input_2)
print('True' if flag else 'False')
