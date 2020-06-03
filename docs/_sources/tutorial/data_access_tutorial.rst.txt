Data access tutorial
====================

Introduction
------------

Wecome to ShgPy! In this tutorial, we will go through the basic steps to understand how ShgPy deals with accessing and manipulating data. Before starting this tutorial, make sure you have installed ShgPy by running::

    $ pip install shgpy

at the terminal. You'll also need to install ``numpy``, ``scipy``, ``sympy``, and, optionally, ``matplotlib``; refer to the corresponding documentation for more information on these packages. If you want to follow along with these tutorials, it's a good idea to download the :doc:`example files <../examples>`.

The main functionality of ShgPy is to be able to fit RA-SHG data to specific fitting formulas, and ultimately to extract susceptibility tensor values from those fits. But before we get into the details, it's useful to first familiarize ourselves with the basic routines for loading, manipulating, and plotting SHG data -- without worrying yet about any internal computation.

Loading RA-SHG data into ShgPy
------------------------------

The first thing we're going to try to do is just to load a collection of RA-SHG data into python. To do this, we're going to make use of some routines in :mod:`shgpy.core.data_handler`. Throughout this tutorial, feel free to reference the API documentation for further details about the functions and classes that we're going to be using.

By the way, this tutorial will be loosely following the example program located at ``examples/data_plot_example.py``, so you are welcome to follow along there (or else type these commands into a python terminal of your own).

The first thing we need to do is to import ``shgpy``:

>>> import shgpy

This gives us access to all of the routines located in the modules :mod:`shgpy.core.data_handler` and :mod:`shgpy.core.utilities`. The function we're going to use right now is :func:`shgpy.core.data_handler.load_data`, which takes two parameters as input. The first is ``data_filenames_dict``, which is a dictionary of filenames labelled by a ``str`` polarization combination. For example, the ``examples/Data`` directory contains four ``.csv`` files

    - ``examples/Data/dataPP.csv``
    - ``examples/Data/dataPS.csv``
    - ``examples/Data/dataSP.csv``
    - ``examples/Data/dataSS.csv``   

which contain RA-SHG data from the (110) surface of GaAs. Take a look at these files to see an example of the type of syntax to use in your own ``.csv`` files.

Working in the ``examples`` directory, let's create a filename dictionary

>>> data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
    'SP':'Data/dataSP.csv',
    'SS':'Data/dataSS.csv',
}

and then call

>>> dat = shgpy.load_data(data_filenames_dict, 'degrees')

The last argument tells ShgPy that the x-axis of our data is typed in degrees rather than radians.

:func:`shgpy.core.data_handler.load_data` returns an instance of a class called :class:`shgpy.core.data_handler.DataContainer`, which does exactly what it says it does -- it contains data. Later we'll see that similar classes exist for holding Fourier-transformed data, formulas, and Fourier-transformed formulas. But for now, let's familiarize ourselves with all the different things we can do with a ``DataContainer`` object.

Note also that in addition to :func:`shgpy.core.data_handler.load_data`, we could have also used :func:`shgpy.core.data_handler.load_data_and_dark_subtract`, if we wanted to dark-subtract our data before loading it in.

First, let's just plot the data to make sure it was accessed correctly. The easiest (but least flexible) way to do this is to use the :mod:`shgpy.plotter` module. Let's import the function ``easy_plot`` from :mod:`shgpy.plotter`

>>> from shgpy.plotter import easy_plot

and then plot the data

>>> easy_plot(
    list_of_dats=[dat],
    list_of_param_dicts=[
        {
            'linestyle':'-',
            'color':'blue',
            'markerfacecolor':'none',
            'marker':'o',
        },
    ],
    pcs_to_include=['PP', 'PS', 'SP', 'SS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)

If you have ``matplolib`` installed correctly, then you should see a (reasonably) nice plot of some RA-SHG data. It's important to note that ShgPy isn't a plotting utility -- in fact, the user is encouraged to write their own routines for making even prettier plots than ``easy_plot`` (e.g. for publication-quality plots). For just checking data and fits however, the routines in :mod:`shgpy.plotter` should do just fine.

For more information about the ``easy_plot`` function, feel free to scroll through the related documentation in :func:`shgpy.plotter.easy_plot`. You'll find that we can also make a polar plot using :func:`shgpy.plotter.easy_polar_plot`.

Now let's see what else we can do with our ``DataContainer`` object. For example, let's write

>>> import numpy as np
>>> dat.scale_data(scale_factor=100)
>>> dat.phase_shift_data(np.pi/2, 'radians')

If we plot the data again, we'll see that all of the data has been scaled by a factor of 100 and rotated through an angle ``np.pi/2``.

Take a minute now to skim the documentation for :class:`shgpy.core.data_handler.DataContainer`, to see what else can be done with ``DataContainer`` s. As always, if there's something you think is missing, feel free to submit a feature request! See :doc:`how to contribute <../contribute>`.

Onec you're satisfied, move on to :doc:`the next tutorial <fitting_tutorial>` to start fitting your data.
