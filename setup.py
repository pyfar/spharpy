#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.22',
    'scipy',
    'urllib3',
    'matplotlib>=3.3.0',
    'pyfar<0.8.0',
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
    author="The pyfar developers",
    author_email='info@pyfar.org',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

    ],
    description="Python package for spherical array processing.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='spharpy',
    name='spharpy',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url="https://pyfar.org/",
    download_url="https://pypi.org/project/spharpy/",
    project_urls={
        "Bug Tracker": "https://github.com/pyfar/spharpy/issues",
        "Documentation": "https://spharpy.readthedocs.io/",
        "Source Code": "https://github.com/pyfar/spharpy",
    },
    version='0.6.1',
    zip_safe=False,
    python_requires='>=3.8',
)
