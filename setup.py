
# Reference: https://stackoverflow.com/questions/7932028/setup-py-for-packages-that-depend-on-both-cython-and-f2py

from setuptools import setup

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(['finsim/portfolio/optimize/native/cythonmetrics.pyx',
                             'finsim/estimate/native/cythonfit.pyx',
                             'finsim/estimate/native/cythonrisk.pyx'])
except ImportError:
    from setuptools import Extension
    ext_modules = [
        Extension('finsim.portfolio.optimize.native.cynthonmetrics',
                  sources=['finsim/portfolio/optimize/native/cythonmetrics.c']),
        Extension('finsim.estimate.native.cythonfit',
                  sources=['finsim/estimate/native/cythonfit.c']),
        Extension('finsim.estimate.native.cythonrisk',
                  sources=['finsim/estimate/native/cythonrisk.c'])
    ]

import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.core import Extension as fortranExtension

fortran_ext_modules = [
    fortranExtension(
        'finsim.portfolio.optimize.native.fortranmetrics',
        sources=['finsim/portfolio/optimize/native/fortranmetrics.f90',
                 'finsim/portfolio/optimize/native/fortranmetrics.pyf']
    ),
    fortranExtension(
        'finsim.estimate.native.fortranfit',
        sources=['finsim/estimate/native/fortranfit.f90',
                 'finsim/estimate/native/fortranfit.pyf']
    ),
    fortranExtension(
        'finsim.estimate.native.fortranrisk',
        sources=['finsim/estimate/native/fortranrisk.f90',
                 'finsim/estimate/native/fortranrisk.pyf']
    ),
    fortranExtension(
        'finsim.simulation.native.f90brownian',
        sources=['finsim/simulation/native/brownian.f90',
                 'finsim/simulation/native/brownian.pyf']
    )
]


def readme():
    with open('README.md') as f:
        return f.read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('## Introduction')
    return text[startpos:]


setup(
    name='finsim',
    version="0.6.10",
    description="Financial simulation and inference",
    long_description=package_description(),
    long_description_content_type='text/markdown',
    classifiers=[
      "Topic :: Scientific/Engineering :: Mathematics",
      "Topic :: Software Development :: Libraries :: Python Modules",
      "Topic :: Software Development :: Version Control :: Git",
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Cython",
      "Programming Language :: C",
      "Programming Language :: Fortran",
      "Intended Audience :: Science/Research",
      "Intended Audience :: Developers",
      "Intended Audience :: Financial and Insurance Industry"
    ],
    keywords="simulation, finance, quantitative finance, inference, portfolio analysis",
    url="https://github.com/stephenhky/FinanceSimulation",
    author="Kwan-Yuet Ho",
    author_email="stephenhky@yahoo.com.hk",
    license='LGPL',
    packages=[
        'finsim',
        'finsim.data',
        'finsim.estimate',
        'finsim.estimate.native',
        'finsim.simulation',
        'finsim.portfolio',
        'finsim.portfolio.optimize',
        'finsim.portfolio.optimize.native'
    ],
    include_dirs=[np.get_include()],
    setup_requires=['Cython', 'numpy', ],
    install_requires=install_requirements(),
    tests_require=[
      'unittest2'
    ],
    # scripts=[],
    include_package_data=True,
    ext_modules=fortran_ext_modules+ext_modules,
    test_suite="test",
    zip_safe=False
)
