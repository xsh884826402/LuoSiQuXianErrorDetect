#32，32 13秒
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import numpy
import time
mod = SourceModule("""
__device__ float getElement(float *a,int row, int col,int width)
{
    return a[row*width+col];
} 
__device__ void setElement(float *c,int row,int col,int width,float value)
{
    c[row*width+col] = value;
}
__global__ void matMulkernel(float *a,float *c)
{
    printf("Hello");
    int width = 128;
    int width1 = 20000;
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    for (int i =0;i<width;++i)
    {
        Cvalue +=getElement(a,row,i,width)*getElement(a,col,i,width);
    }
    setElement(c,row,col,width1,Cvalue);

}
""")
start = time.time()
a = numpy.random.randn(20000,128)
a = a.astype(numpy.float32)
print('nbytes',a.nbytes,a.shape)
c = numpy.zeros((20000,20000),dtype=numpy.float32)
print('c nbytes',c.nbytes)
a_gpu = cuda.mem_alloc(a.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu,a)
# cuda.memcpy_htod(c_gpu,c)


func = mod.get_function("matMulkernel")
threadsPerBlock = (32,32,1)
grid =(20000//32,20000//32)
width = '128'
# width_gpu = cuda.mem_alloc(sys.getsizeof(width))
# cuda.memcpy_htod(width_gpu,width)
func(a_gpu,c_gpu,grid=grid,block=threadsPerBlock)
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(c,c_gpu)
print(c.shape)
print(a.shape)
print("time",time.time()-start)
print("c末尾",c[-1][-1],numpy.dot(a[-1],a[-1]))
index = 0
for i in range(20000):
    if c[i][0] ==0:
        print('Here')
        index = i
        break
print(index,c[index-1],c[index],c[index+1])