
from setuptools import setup

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
    version="0.2.0",
    description="Financial simulation and inference",
    long_description=package_description(),
    long_description_content_type='text/markdown',
    classifiers=[
      "Topic :: Scientific/Engineering :: Mathematics",
      "Topic :: Software Development :: Libraries :: Python Modules",
      "Topic :: Software Development :: Version Control :: Git",
      "Programming Language :: Python :: 3.6",
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
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
        'finsim.portfolio'
    ],
    install_requires=install_requirements(),
    tests_require=[
      'unittest2', 'pytest'
    ],
    # scripts=[],
    include_package_data=True,
    test_suite="test",
    zip_safe=False
)
