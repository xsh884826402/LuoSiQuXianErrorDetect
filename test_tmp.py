import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
print(tools.DeviceData(dev = cuda.Context.get_device()).max_threads)