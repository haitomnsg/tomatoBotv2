from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os # <--- ADD THIS LINE

# Get the directory of the current setup.py file
this_dir = os.path.dirname(os.path.abspath(__file__)) # <--- ADD THESE LINES

setup(
    name='deep_hough',
    ext_modules=[
        CUDAExtension('deep_hough', [
            # Use os.path.join to create correct relative paths
            os.path.join(this_dir, 'deep_hough_cuda.cpp'), # <--- MODIFY THIS LINE
            os.path.join(this_dir, 'deep_hough_cuda_kernel.cu'), # <--- MODIFY THIS LINE
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })