#458s,503s c[i][j] =numpy.dot(a[i],a[j])使用矩阵乘法
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import numpy
import time

start = time.time()
a = numpy.random.randn(20000,128)
a = a.astype(numpy.float32)
c = numpy.zeros((20000,20000),dtype=numpy.float32)
for i in range (20000):
    for j in range(20000):
        c[i][j] = numpy.dot(a[i],a[j])

print(c.shape,c[0][0])
print(a.shape,a[0])
print("time",time.time()-start)