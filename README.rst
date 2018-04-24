=======
SPHARPY
=======


Package for SPherical ARray processing in PYthon.

===============
Getting Started
===============

There is no stable release and therefore no binary distribution yet. Hence, the
package needs to be installed from source.
When installing from source, we need to compile some C++ code using Cython. This will be handled automatically, but may introduce an additional step.

Requirements
============

- Boost C++ Libraries
- C++ Compiler (MSVC, gcc tested)
- Python (>= 3.5 recommended)
- Python packages: numpy, scipy, matplotlib, urllib3, cython

Installation
============

The sources for spharpy can be downloaded from the `git repository`_.

You can either clone the public repository

.. code-block:: console

    $ git clone git://git.rwth-aachen.de/mbe/spharpy

or download the code from the `GitLab pages`_.

Once you have a copy of the source and the Boost libraries are located in your
systems PATH, you can install it with:

.. code-block:: console

    $ python setup.py install

The required packages to install the package can be installed using pip:

.. code-block:: console

    $ pip install -r requirements.txt

If the Boost libraries are not located in your systems PATH, you have to tell
the setup function where to find them using:

.. code-block:: console

    $ python setup.py build_ext --inplace --include=/path/to/boost_1_xx/
    $ python setup.py install

When actively developing for the package it is recommended to install using the
develop option:

.. code-block:: console

    $ python setup.py develop

The packages required for development, building the documentation, etc. can be installed using:

.. code-block:: console

    $ pip install -r requirements_dev.txt


.. _GitLab pages: https://git.rwth-aachen.de/mbe/spharpy
.. _git repository: https://git.rwth-aachen.de/mbe/spharpy


=============
Documentation
=============

The documentation files are located inside the docs folder. The documentation
can be generated using sphinx by running:

.. code-block::

    $ cd docs/
    $ make html

Make sure that the Cython extension modules have been build previous to building
the documentation. The extension modules are built when installing the package.
They can however be built without installing by running:

.. code-block::

    $ python setup.py build_ext --inplace
