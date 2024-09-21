
from setuptools import setup
import numpy as np
from Cython.Build import cythonize

ext_modules = cythonize(['finsim/portfolio/optimize/native/cythonmetrics.pyx',
                         'finsim/estimate/native/cythonfit.pyx',
                         'finsim/estimate/native/cythonrisk.pyx'])


setup(
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,
)
