from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cartpole2l_cpp',
    ext_modules=[
        CppExtension('cartpole2l_cpp', 
            sources = ['forward_dynamics_wrapper.cpp', 'forward_dynamics.c'],
            extra_compile_args={
                'cxx':  ['-O3'], 
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
