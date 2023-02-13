#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The setup script.
"""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.22',
    'scipy',
    'urllib3',
    'matplotlib>=3.3.0'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'wheel',
    'flake8',
    'bump2version',
]


setup(
    name='spharpy',
    version='0.4.2',
    description="Python package for spherical array processing.",
    long_description=readme,
    author="Marco Berzborn",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    url='https://github.com/mberz/spharpy',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='spharpy',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    download_url="https://pypi.org/project/spharpy/",
    project_urls={
        "Bug Tracker": "https://github.com/mberz/spharpy/issues",
        "Documentation": "https://spharpy.readthedocs.io/",
        "Source Code": "https://github.com/mberz/spharpy",
    },
    python_requires='>=3.8'
)
