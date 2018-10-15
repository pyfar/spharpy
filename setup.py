#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script.
The package uses Cython extension modules. Parallelization using OpenMP is
currently only supported on Linux using gcc.
"""

import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.10',
    'cython',
    'scipy',
    'urllib3',
    'matplotlib'
]

setup_requirements = [
    'pytest-runner',
    'cython'
]

test_requirements = [
    'pytest',
    'cython'
]

if sys.platform.startswith('linux'):
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']
else:
    compile_args = ['/openmp']
    link_args = []


spherical_ext = Extension(name="spharpy.spherical",
                          sources=["./spharpy/spherical_ext/spherical.pyx"],
                          language="c++",
                          extra_compile_args=compile_args,
                          extra_link_args=link_args,
                          include_dirs=[numpy.get_include(),
                                        "./spharpy/spherical_ext/",
                                        "./spharpy/special/"])

special_ext = Extension(name="spharpy.special",
                        sources=["./spharpy/special/special.pyx"],
                        language="c++",
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[numpy.get_include(),
                                      "./spharpy/special/"])


setup(
    name='spharpy',
    version='0.1.2',
    description="Python package for spherical array processing.",
    long_description=readme + '\n\n' + history,
    author="Marco Berzborn",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    url='https://git.rwth-aachen.de/mbe/spharpy/',
    packages=find_packages(),
    package_data = {
        'spharpy/special/_special': ['spharpy/special/_special.pxd']
        },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='spharpy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    ext_modules=cythonize([spherical_ext, special_ext])
)
