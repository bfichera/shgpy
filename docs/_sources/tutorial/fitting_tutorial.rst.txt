Fitting tutorial
================

Prerequisites
-------------

Before going through this tutorial, make sure you've :doc:`installed shgpy <../index>` and read through :doc:`the last tutorial <tensor_tutorial>`.

Introduction
------------

In :doc:`the first tutorial <data_access_tutorial>`, we learned how to load RA-SHG data into ShgPy using :func:`shgpy.core.data_handler.load_data` and the `shgpy.core.data_handler.DataContainer` class. In :doc:`the last tutorial <tensor_tutorial>`, we learned about how tensors for different point groups were defined in ShgPy and how to manipulate them. Now, we're going to put these concepts together and learn how to fit RA-SHG data.

Fourier formula generation
--------------------------

As alluded to previously, the central idea behind fitting in ShgPy is to fit in Fourier space. This provides a drastic simplification to the cost function. However, the problem is that computing a Fourier transform symbolically is difficult, and we have resort to some tricks to compute it efficiently (or at least, ahead of time).

What do I mean by the last part? To begin, let's think about what the function is that we're trying to compute. Ultimately, we want to compute an intensity as a function of the azimuthal angle ``phi`` in the experiment. This is given by the square of the nonlinear polarization, i.e.::

    I = |P_i|**2 = |chi_ijk E_j E_k|**2

What part of this formula depends on ``phi``? In the experiment, the electric field changes as a function of ``phi`` like::

    E_i(phi) = R_ij(phi) E_j

And that's it -- no other part of the formula depends on ``phi`` (note: it's actually more complicated than this; in code we not only consider an additional quadrupole contribution, but also the fact that the component of the SHG signal along the direction of propogation is not measurable. However, these considerations do not affect the basic argument here; feel free to look through the source code of :func:`shgpy.fformgen.generate_uncontracted_fourier_transforms` for more information).

In particular, the susceptibility tensor , which is the only part of the formula that will change from problem to problem, does not natively depend on ``phi``. Therefore, to compute the Fourier transform of the intensity, we can compute the Fourier transform of everything not involving the susceptibility, and then do a (conceptually complicated, but not numerically difficult) contraction by ``chi_ijk``. In ShgPy, we perform this two-step process by

1. Running :func:`shgpy.fformgen.generate_uncontracted_fourier_transforms`
2. Running :func:`shgpy.fformgen.generate_contracted_fourier_transforms`

Most importantly, since step 1 involves every part of the formula which doesn't depend on ``chi``, it only needs to be run once. The result can then be cached and used every time you want to calculate a new Fourier formula (e.g. because you want to fit a new tensor). Step 2 is more specific, but only has to be run once for each tensor you want to try to fit. The result can then be saved and used later, having saved a lot of computation time.

That all was pretty conceptual, but luckily, none of the details are really important in order to *use* ShgPy (note: if there's interest, I would be happy to expand more on this point; see :doc:`how to contribute <../contribute>`). For now, let's just see how it all works in practice.

Remember that the goal is to generate a formula for the SHG intensity as a function of ``phi`` (or, since we're working in Fourier space, a Fourier formula for the SHG intensity as a function of the Fourier frequency ``n``). We proceed according to steps 1 and 2 above.

To perform step 1, let's follow ``examples/generate_uft_example.py``. We start by importing the logging module, which provides a flexible event-logging system and is widely implemented in ShgPy.

>>> import logging

We'll also need the :mod:`shgpy.core` modules and :mod:`shgpy.fformgen`:

>>> import shgpy
>>> import shgpy.fformgen

Let's configure the logger:

>>> mylogger = logging.getLogger(__name__)
>>> logging.basicConfig(level=logging.DEBUG)

(Note that while useful, the logging implementation is purely optional; it just let's us look into some of the debugging messages produced by the functions in :func:`shgpy.fformgen`).

Although the angle of incidence can be left as a free variable in the Fourier formula generation (see :func:`shgpy.fformgen.generate_uncontracted_fourier_transforms_symb` and ``examples/generate_uft_symb_examples.py``), it is a useless complication unless truly needed. So let's hardcode it:

>>> AOI = 0.1745  # 10 degrees, in radians

For your implementation, you may want to use a different angle of incidence.

Now we're ready to generated the uncontracted Fourier transforms. Simply run

>>> shgpy.fformgen.generate_uncontracted_fourier_transforms(AOI, 'uft_filename_prefix')

If you configured ``logging``, you should start to see a bunch of debug messages start to print out (they're mostly meaningless, but at least you know that something's going on). This calculation takes about five minutes on my machine. Note here that 'uft_filename_prefix' is a prefix to the paths where you want to save the cached answers. In the examples, we make a directory ``examples/uft`` and save the answers at ``examples/uft/uft10deg``. That means that :func:`shgpy.fformgen.generate_uncontracted_fourier_transforms` will save four files: ``examples/uft/uft10deg_pp.p``, ``examples/uft/uft10deg_ps.p``, ``examples/uft/uft10deg_sp.p``, and ``examples/uft/uft10deg_ss.p``, each of which corresponds to a particular uncontracted Fourier transform.

Note that in the typical use case, the above should be the only time you have to run :func:`shgpy.fformgen.generate_uncontracted_fourier_transforms`. The answers saved at ``'uft_filename_prefix'+...`` can be used for essentially any SHG fitting problem that you might encounter.

Now let us turn to our specific use case. As an example, imagine that we are trying to fit the GaAs data available in ``examples/Data`` to the tensor ``shgpy.tensor_definitions.dipole['T_d']`` oriented along the (110) direction. First, we define the fitting tensor

>>> from shgpy.tensor_definitions import dipole
>>> t_dipole = shgpy.particularize(dipole['T_d'])
>>> import numpy as np
>>> R = shgpy.rotation_matrix_from_two_vectors(
    np.array([1, 1, 0]),
    np.array([0, 0, 1]),
)
>>> t_dipole = shgpy.transform(t_dipole, R)
>>> t_dipole = shgpy.make_tensor_real(_)

We're not going to add any quadrupole contribution, so we can set the quadrupole tensor to zero:

>>> import sympy
>>> t_quad = np.zeros(shape=(3,3,3,3), dype=sympy.Expr)

Lastly, we'll define the place that we want to save the Fourier formula

>>> save_filename = 'T_d-None-None(110)-particularized.p'

(Note: this is the typical filename convention for Fourier formulas. It denotes the dipole, surface, and quadrupole tensors used, the orientation, and the fact that the tensor was particularized.)

Finally, we run

>>> shgpy.fformgen.generate_contracted_fourier_transforms(save_filename, 'uft/uft10deg', t_dipole, t_quad, ndigits=4)

On my machine, this takes about five to ten minutes, depending on the complexity of the susceptibility tensors. When it completes, the function will save a pickled Fourier formula object to the location specified by ``save_filename``.

What we've just done is by far the most difficult step (both conceptually and computationally) in ShgPy, but it is easily worth it. By spending 10-15 minutes of computation time now, we have dramatically simplified the routines that we are about to run in the next section of this tutorial.

The final step: fitting your first RA-SHG data
----------------------------------------------   

All that's left now is to load the Fourier formula just generated (at ``'T_d-None-None(110)-particularized.p'``) into ShgPy, load the data that we want to fit, and then fun one of the functions in :mod:`shgpy.fformfit`.

Before we begin, let's recall from :doc:`the first tutorial <data_access_tutorial>` how we loaded RA-SHG data into ShgPy. In that tutorial, we loaded the data into an instance of the special class :class:`shgpy.core.data_handler.DataContainer`, and noted that other datatypes would be loaded into similar objects when it came to actually doing the fitting.

Let's review these other datatypes now. First, we consider the class :class:`shgpy.core.data_handler.fDataContainer`, which, in brief, simply contains the Fourier transform of the sort of data which would go into a :class:`shgpy.core.data_handler.DataContainer` instance. Like :class:`shgpy.core.data_handler.DataContainer`, it also includes methods for scaling and phase-shifting the data contained in it.

To create an instance of :class:`shgpy.core.data_handler.fDataContainer`, one can load a dataset into a :class:`shgpy.core.data_handler.DataContainer` instance and then convert it using :func:`shgpy.core.data_handler.dat_to_fdat`, or use the function :func:`shgpy.core.data_handler.load_data_and_fourier_transform`, which does both at the same time:

>>> data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
    'SP':'Data/dataSP.csv',
    'SS':'Data/dataSS.csv',
}
>>> dat, fdat = shgpy.load_data_and_fourier_transform(data_filenames_dict, 'degrees')

Ultimately, it is the data contained in an :func:`shgpy.core.data_handler.fDataContainer` object that we are going to want to fit to.

The fitting formula, on the other other hand, is stored in a related object called :class:`shgpy.core.data_handler.fFormContainer`. To create an instance of :class:`shgpy.core.data_handler.fFormContainer`, simply load the Fourier formula we just created

>>> fform_filename = 'T_d-None-None(110)-particularized.p'
>>> fform = shgpy.load_fform(fform_filename)

This would be a good time to read the documentation provided in :mod:`shgpy.core.data_handler` to familiarize oneself with these functions. (You will find that there is a fourth object, :class:`shgpy.core.data_handler.FormContainer`, which is designed to contain ``phi``-space formulas; see also :mod:`shgpy.formgen` and the documentation therein for more details.)

There is one more fitting parameter which is not captured by :func:`shgpy.fformgen.generate_contracted_fourier_transforms`, which is the relative phase shift between the data and the fitting formula. So let's phase shift the formula by an arbitrary angle.

>>> from shgpy.shg_symbols import psi
>>> fform.apply_phase_shift(psi)

The fitting routines require an initial guess; let's just guess 1 for each parameter:

>>> guess_dict = {}
>>> for fs in fform.get_free_symbols():
>>>     guess_dict[fs] = 1

And now we're finally ready to run the fitting:

>>> from shgpy.fformfit import least_squares_fit
>>> ret = least_squares_fit(fform, fdat, guess_dict)

Here, ``ret`` is an instance of the `scipy.optimize.OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`_ class, see the documentation in that link for more information. The most important attribute of ``ret`` for us is the answer:

>>> ret.xdict
{psi: 1.5914701873213561, zyx: 1.2314580678986173}

In addition to :func:`shgpy.fformfit.least_squares_fit`, there are a couple of other routines available for fitting RA-SHG data. The most useful one for most problems is actually :func:`shgpy.fformfit.basinhopping_fit` (and its cousins, see the :mod:`shgpy.fformfit` reference), which is based on the `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping>`_ function provided by SciPy. It is specifically designed to treat problems with many local minima and degrees of freedom. In the future, further fitting routines will be added, if there is interest (see :doc:`how to contribute <../contribute>`).

A variant of the basinhopping algorithm which is also included in :mod:`shgpy.fformfit` is :func:`shgpy.fformfit.dual_annealing_fit`. See the API documentation for more information.

Before concluding this tutorial, let me add one more comment about one important capability of this software. Once the fitting routine has finished generating the appropriate energy cost expression using ``fform`` and ``fdat``, it turns it into C code using ``sympy.utilities.codegen`` and compiles a shared object file, which it runs using ``ctypes`` during the fitting process. This drastically reduces computation time for complicated fitting functions, for which I've found ``sympy.lambdify`` to be extremely slow. As a result, if you want to save the generated shared object file and then load it for the next simulation, you can use the ``save_cost_func_filename`` and ``load_cost_func_filename`` options (and those related to them) in the fitting routines of :mod:`shgpy.fformfit`.

If you'd like to generate the cost function without running the fitting routine directly afterwards (as opposed to running them in series, which, for backwards-compatibility, is what the aforementioned :mod:`shgpy.fformfit` routines do), use :func:`shgpy.fformfit.gen_cost_func`.

Conclusion
----------

This concludes the ShgPy tutorials. For more information, I recommend looking through the :doc:`API <../modules>`; there are a lot of important functions there which we haven't covered here but may be useful for your application. And, as always, if you have questions please feel free to :doc:`contact me <../contact>`.
