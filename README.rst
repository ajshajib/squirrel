==========================================
🐿️ squirrel
==========================================

|Read the Docs| |GitHub| |Codecov| |Black| |docformatter| |docstyle|

`squirrel` is a Python package for kinematic measurements, designed to simplify the fitting of multiple kinematic 
components at different redshifts, such as those found in the spectra of strongly lensed systems. Built on top of 
pPXF, `squirrel` offers quality-of-life features that streamline the kinematic fitting workflow, making it easier to 
analyze spectra with both single and multiple kinematic components.

Installation
============

To install squirrel from source, first clone the repository:

.. code-block:: bash

    git clone https://github.com/ajshajib/squirrel.git
    cd squirrel

Install the package and its dependencies:

.. code-block:: bash

    pip install -r requirements.txt
    pip install .

To install development dependencies (for testing, docs, etc.):

.. code-block:: bash

    pip install -r requirements_dev.txt

You can now import and use squirrel in your Python scripts or notebooks.

For more details, see the documentation: https://squirrel.readthedocs.io/en/latest/

Citation
========

`squirrel` was developed in `Shajib et al. (2025) <https://ui.adsabs.harvard.edu/abs/2025arXiv250621665S/abstract>`_. 
Please cite this paper if you use `squirrel` in your research. If you also use the methodolgy of 
`Knabel, Mozumdar, et al. (2025) <https://ui.adsabs.harvard.edu/abs/2025arXiv250216034K/abstract>`_ 
to combine mulitple template libraries, please cite that paper as well.

.. |Read the Docs| image:: https://readthedocs.org/projects/squirrel-docs/badge/?version=latest
    :target: https://squirrel-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |GitHub| image:: https://github.com/ajshajib/squirrel/workflows/CI/badge.svg
    :target: https://github.com/ajshajib/squirrel/actions
    :alt: Build Status

.. |Codecov| image:: https://codecov.io/gh/ajshajib/squirrel/graph/badge.svg?token=PyDRdtsGSX
    :target: https://codecov.io/gh/ajshajib/squirrel
    :alt: Code coverage

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |docstyle| image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
    :target: https://www.sphinx-doc.org/en/master/usage/index.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter
