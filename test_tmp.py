import pyemd
import  numpy as np
a =[1,2,3]
b = [4,5,6]
a = np.array(a)
b = np.array(b)
res = pyemd.emd_samples(a,b)
print(res,type(res))
