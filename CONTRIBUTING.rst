============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/mberz/spharpy/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

spharpy could always use more documentation, whether as part of the
official spharpy docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/mberz/spharpy/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up spharpy for local development.

1. Fork the spharpy repo on GitHub.
2. Clone your fork locally

.. code-block:: console

    $ git clone https://github.com/mberz/spharpy.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development

.. code-block:: console

    $ mkvirtualenv spharpy
    $ cd spharpy/
    $ python setup.py develop

4. Create a branch for local development. Indicate the intention of your branch in its
   respective name (i.e. ``feature/branch-name`` or ``bugfix/branch-name``)

.. code-block:: console

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox

.. code-block:: console

    $ flake8 spharpy tests
    $ pytest
    $ tox

To get flake8 and tox, pip install them into your virtualenv. The flake8 test must pass without any
warnings for ``./spharpy`` and ``./tests`` using the default or a stricter configuration.
Flake8 ignores `E123/E133, E226` and `E241/E242` by default. If necessary adjust
your flake8 and linting configuration in your IDE accordingly.

6. Commit your changes and push your branch to GitHub

.. code-block:: console

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for Python >=3.7. Make sure that the tests pass for all supported Python versions.


Testing Guidelines
-----------------------
Spharpy uses test-driven development based on
`three steps <https://martinfowler.com/bliki/TestDrivenDevelopment.html>`_ and
`continuous integration <https://en.wikipedia.org/wiki/Continuous_integration>`_ to test and monitor the code.
In the following, you'll find a guideline. Note: these instructions are not generally applicable outside of spharpy.

- The main tool used for testing is `pytest <https://docs.pytest.org/en/stable/index.html>`_.
- All tests are located in the *tests/* folder.
- Make sure that all important parts of spharpy are covered by the tests.
  This can be checked using *coverage* (see below).
- In case of spharpy, mainly **state verification** is applied in the tests.
  This means that the outcome of a function is compared to a desired value (``assert ...``).
  For more information, it is referred to `Martin Fowler's article <https://martinfowler.com/articles/mocksArentStubs.html.>`_.


Tips
~~~~~~~~~~~
Pytest provides several, sophisticated functionalities which could reduce the effort of implementing tests.

- Similar tests executing the same code with different variables can be
  `parametrized <https://docs.pytest.org/en/stable/example/parametrize.html>`_.
- Feel free to add more recommendations on useful pytest functionalities here.
  Consider, that a trade-off between easy implementation and good readability of the tests needs to be found.

You can create an html report on the test `coverage <https://coverage.readthedocs.io/en/coverage-5.5/>`_ by calling

.. code-block:: console

    $ pytest --cov=. --cov-report=html


Writing the Documentation
-------------------------

Spharpy follows the `numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for the docstring.
A docstring has to consist at least of

- A short and/or extended summary,
- the Parameters section, and
- the Returns section

Optional fields that are often used are

- References,
- Examples, and
- Notes

Here are a few tips to make things run smoothly

- Use the tags ``:py:func:``, ``:py:mod:``, and ``:py:class:`` to reference functions, modules, and classes: For example ``:py:func:`~spharpy.plot.balloon``` for a link that displays only the function name.
- Code snippets and values as well as external modules, classes, functions are marked by double ticks \`\` to appear in mono spaced font, e.g., ``x=3`` or ``spharpy.transforms.RotationSH``.
- Parameters, returns, and attributes are marked by single ticks \` to appear as emphasized text, e.g., *unit*.
- Use ``[#]_`` and ``.. [#]`` to get automatically numbered footnotes.
- Do not use footnotes in the short summary. Only use footnotes in the extended summary if there is a short summary. Otherwise, it messes with the auto-footnotes.
- Plots can be included in by using the prefix ``.. plot::`` followed by an empty line and an indented block containing the code for the plot. See `spharpy.plot` for examples.

See the `Sphinx homepage <https://www.sphinx-doc.org>`_ for more information.

Building the Documentation
--------------------------

You can build the documentation of your branch using Sphinx by executing the make script inside the docs folder.

.. code-block:: console

    $ cd docs/
    $ make html

After Sphinx finishes you can open the generated html using any browser

.. code-block:: console

    $ docs/_build/index.html

Note that some warnings are only shown the first time you build the
documentation. To show the warnings again use

.. code-block:: console

    $ make clean

before building the documentation.


Deploying
~~~~~~~~~

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run

.. code-block:: console

    $ bumpversion patch # possible: major / minor / patch
    $ git push
    $ git push --tags

CircleCI will then deploy to PyPI if tests pass.
