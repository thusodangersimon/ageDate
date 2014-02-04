#!/usr/bin/env python2

"""used to make wrapper from boost c++ code"""
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("losvd_convolve", ["losvd_convolve.cpp"],
        libraries = ["boost_python"])
    ])
