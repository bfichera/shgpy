.. ShgPy documentation master file, created by
   sphinx-quickstart on Sun May 31 23:51:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Welcome to ShgPy!
=================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3873755.svg
   :target: https://doi.org/10.5281/zenodo.3873755

What is ShgPy?
==============

ShgPy is a simple toolkit for analyzing rotational-anisotropy second harmonic generation (RA-SHG) data. It depends mainly on three packages, `NumPy <https://numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, and `SymPy <https://www.sympy.org/en/index.html>`_ -- as well as (optionally) `Matplolib <https://matplotlib.org/>`_ for some basic plotting capability -- to simulate, manipulate, and fit RA-SHG data in a (hopefully!) intuitive way.

Fitting RA-SHG data involves solving a complex global minimization problem with many degrees of freedom. Moreover, the fitting functions naively involve a degree of complexity due to the trigonometric nature of the problem. This software therefore takes a dual approach to fitting RA-SHG data -- first of all, all of the fits are done in Fourier space, which significantly reduces the complexity of the fitting formulas, and second of all, ShgPy makes heavy use of the `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_ algorithm, which is particularly useful for these types of global optimization problems. In some ways, ShgPy is just a wrapper around `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html#optimization-and-root-finding-scipy-optimize>`_.



What ShgPy is not
=================

Right now, ShgPy is narrowly suited for a particular RA-SHG geometry (i.e. `this one <https://arxiv.org/abs/1909.12850>`_). Other implementations of RA-SHG (e.g. where the angle of incidence can vary, etc.) are not currently supported. However, this software is always evolving and support may come in the future if there's enough interest (see :doc:`how to contribute <contribute>`).

Installation
============

Installation of ShgPy is easy! Just install the ``shgpy`` package::

    $ pip install shgpy

You'll also need to install ``numpy``, ``scipy``, ``sympy``, and, optionally, ``matplotlib``; refer to the corresponding documentation for more information about these packages. Additionally, you'll need a working installation of ``gcc``.

Getting started
===============

After installing ShgPy, familiarize yourself with the basic routines by reading the :doc:`tutorials <tutorial/tutorials>` and :doc:`FAQs <faq>`.

.. toctree::
   :maxdepth: 2

   tutorial/tutorials

To follow along, you'll want to download the :doc:`example files <examples>`.

.. toctree::
   :hidden:

   examples 

.. toctree::
   :maxdepth: 1
   :hidden:

   faq

Documentation
=============

For further details, consult the auto-generated API documentation:

.. toctree::

   modules

.. toctree::
   :maxdepth: 1
   :hidden:

   changelog

How to contribute
=================

See :doc:`how to contribute <contribute>`.

.. toctree::
   :hidden:

   contribute

Cite this software
==================

If you use ShgPy in your project, please :doc:`let me know <contact>`! I'm interested to see who is using this software and in what form.

To cite this software in a publication, use

- Bryan Fichera. *ShgPy*. `<https://bfichera.github.io/shgpy/>`_, 2020. `doi:10.5281/zenodo.3873755 <https://doi.org/10.5281/zenodo.3873755>`_

Alternatively, use the following BibTex entry::

    @misc{shgpy,
        author = {Bryan Fichera},
        title = {Shg{P}y},
        year = {2020},
        howpublished = {\url{https://bfichera.github.io/shgpy/}},
        doi = {10.5281/zenodo.3873755}
    }


Contact me
==========

Other questions or concerns? Please :doc:`contact me <contact>`!

.. toctree::
   :hidden:

   contact

   




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
