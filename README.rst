=======
SPHARPY
=======

Package for SPherical ARray processing in PYthon.

===============
Getting Started
===============

There is no stable release and therefore no binary distribution yet. Hence, the
package needs to be installed from source.

Requirements
============

- Python (>= 3.5 recommended)
- Python packages: numpy, scipy, matplotlib, urllib3

Installation
============

The sources for spharpy can be downloaded from the `git repository`_.

You can either clone the public repository

.. code-block:: console

    $ git clone git://git.rwth-aachen.de/mbe/spharpy

or download the code from the `GitLab pages`_.

You can install it with:

.. code-block:: console

    $ python setup.py install

The required packages to install the package can be installed using pip:

.. code-block:: console

    $ pip install -r requirements.txt

When actively developing for the package it is recommended to install using the
develop option:

.. code-block:: console

    $ python setup.py develop

The packages required for development, building the documentation,
etc. can be installed using:

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
