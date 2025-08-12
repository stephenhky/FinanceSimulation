Welcome to finsim's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Introduction
============

Financial assets estimation, and portfolio management.

Installation
============

To install finsim, use pip:

.. code-block:: bash

   pip install finsim

Usage
=====

.. code-block:: python

   from finsim.data import FinnHubStockReader
   from finsim.estimate import fit_BlackScholesMerton_model
   from finsim.portfolio import get_optimized_portfolio_on_sharpe_ratio

   # Example usage
   # ...

API Reference
=============

.. toctree::
   :maxdepth: 4

   finsim