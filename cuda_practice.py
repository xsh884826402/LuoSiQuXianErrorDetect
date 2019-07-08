import numpy as np
from timeit import default_timer as timer
import numba as na
#16秒
@na.vectorize(["float32(float32,float32)"],target='cuda')
def vectorMul(a, b):

    # res = 0
    # for i in range(len(c)):
    #     res +=c[i]
    # return res
    return  a*b

def main():
    N = 20000
    A = np.random.random_sample((N,128))
    B = np.random.random_sample((N,128))
    A = np.array(A,dtype='float32')
    B = np.array(B,dtype='float32')
    C = np.zeros((N,N), dtype='float32' )
    print(type(A[0]))
    start = timer()
    for i in range(N):
        for j in range(N):
            C[i][j] = np.sum(vectorMul(A[i],B[j]))
    # C = vectorAdd(A, B)
    vectorAdd_time = timer() - start
    print("C_shape",C.shape)
    print("c[:5] = " + str(C[:5]))
    print("c[-5:] = " + str(C[-5:]))

    print("vectorAdd took %f seconds " % vectorAdd_time)

if __name__ == '__main__':
    main()
    # print(os.environ['NUMBAPRO_NVVM'])
# export NUMBAPRO_NVVM=/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so
# export NUMBAPRO_LIBDEVICE=/usr/local/cuda-8.0/nvvm/libdevice/