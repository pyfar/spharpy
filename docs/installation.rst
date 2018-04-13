.. highlight:: shell

============
Installation
============


Stable release
--------------

There is not stable release yet. See next section on installing from sources.




From sources
------------

When installing from sources, we need to compile some C++ code using Cython. This will be handled automatically, but may introduce an additional step.

Requirements
============

- Boost C++ Libraries
- C++ Compiler
- Python (>= 3.5 recommended)
- Python packages: numpy, scipy, matplotlib, urllib3, cython

Installation
============

The sources for spharpy can be downloaded from the `git repository`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://git.rwth-aachen.de/mbe/spharpy

or download the code from the `GitLab pages`_.

Once you have a copy of the source and the Boost libraries are located in your
systems PATH, you can install it with:

.. code-block:: console

    $ python setup.py install

If the Boost libraries are not located in your systems PATH, you have to tell
the setup function where to find them using:

.. code-block:: console

    $ python setup.py build_ext --inplace --include=/path/to/boost_1_xx/
    $ python setup.py install

.. _GitLab pages: https://git.rwth-aachen.de/mbe/spharpy
.. _git repository: https://git.rwth-aachen.de/mbe/spharpy
