.. ShgPy documentation master file, created by
   sphinx-quickstart on Sun May 31 23:51:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Welcome to ShgPy!
=================

What is ShgPy?
==============

ShgPy is an open-source toolkit for analyzing rotational-anisotropy
second harmonic generation (RA-SHG) data. It depends mainly on
three packages, `NumPy <https://numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, and `SymPy <https://www.sympy.org/en/index.html>`_ -- as well as (optionally) `Matplolib <https://matplotlib.org/>`_ for some basic plotting capability -- to simulate, manipulate, and fit RA-SHG data in a (hopefully!) intuitive way.

Fitting RA-SHG data involves solving a complex global minimization problem with many degrees of freedom. Moreover, the fitting functions naively involve a degree of complexity due to the trigonometric nature of the problem. This software therefore takes a dual approach to fitting RA-SHG data -- first of all, all of the fits are done in Fourier space, which significantly reduces the complexity of the fitting formulas, and second of all, ShgPy makes heavy use of the `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_ algorithm, which is particularly useful for these types of global optimization problems.

What ShgPy is not
=================

Right now, ShgPy is narrowly suited for a particular RA-SHG geometry (i.e. `this one <https://arxiv.org/abs/1909.12850>`_). Other implementations of RA-SHG (e.g. where the angle of incidence can vary, etc.) are not currently supported. However, this software is always evolving and support may come in the future if there's enough interest (see :doc:`contribute`).

Installation
============

Installation of ShgPy is easy! Just install the `shgpy` package::

    $ pip install shgpy

Getting started
===============

After installing ShgPy, the next thing to do is to read the tutorials.

Documentation
=============

Know what you're looking for and just want API details? View the auto-generated API documentation:

.. toctree::
   :maxdepth: 4

   modules

How to contribute
=================

.. toctree::
   :maxdepth: 2

   contribute



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
