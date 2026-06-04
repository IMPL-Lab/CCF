from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='iou3d_cuda',
    ext_modules=[
        CUDAExtension(
            name='iou3d_cuda',
            sources=[
                'src/iou3d.cpp',
                'src/iou3d_kernel.cu'
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)