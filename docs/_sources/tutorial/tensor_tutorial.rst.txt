Tensor tutorial
===============

Prerequisites
-------------

Before going through this tutorial, make sure you've :doc:`installed shgpy <../index>` and read through :doc:`the last tutorial <data_access_tutorial>`.

Introduction
------------

RA-SHG is a technique which is designed to measure a particular set of numbers which we collectively call the "susceptibility tensor." This tutorial will go through the basics of how tensors are implemented in ShgPy, so that in the next tutorial we'll know how to generate fitting formulas depending on the tensor that we want to fit to.

Tensor definitions
------------------

Depending on the point group of the material that we're trying to study, the susceptibility tensor will take on a variety of different forms. For example, if the material has inversion symmetry, then the susceptibility tensor will be identically zero -- if, instead, it has threefold rotational symmetry, it might take on a form like::

    chi = [[[ xxx, -yyy,  yyz],
            [-yyy, -xxx, -yxz],
            [ yzy, -yzx,   0 ]],

           [[-yyy, -xxx,  yxz],
            [-xxx,  yyy,  yyz],
            [ yzx,  yzy,   0 ]],

           [[zyy,  -zyx,   0 ],
            [zyx,   zyy,   0 ],
            [ 0 ,    0,   zzz]]]

In ShgPy, all of these definitions (e.g. for each of the 32 crystallographic point groups) are defined in :mod:`shgpy.tensor_definitions`. :mod:`shgpy.tensor_definitions` defines three dictionaries: ``dipole``, ``surface``, and ``quadrupole``. Let's look at what these dictionaries contain.

First, let's import the dictionary ``dipole``:

>>> from shgpy.tensor_definitions import dipole

and look at its keys:

>>> dipole.keys() 
dict_keys(['S_2', 'C_2h', 'D_2h', 'C_4h', 'D_4h', 'T_h', 'O_h', 'S_6', 'D_3d', 'C_6h', 'D_6h', 'C_2', 'C_1h', 'D_2', 'C_2v', 'C_4', 'S_4', 'D_4', 'C_4v', 'D_2d', 'O', 'T_d', 'T', 'D_3', 'C_3', 'C_3v', 'C_6', 'C_3h', 'D_6', 'C_6v', 'D_3h', 'C_1'])

so the ``dipole`` dictionary contains one entry for each of the 32 crystallographic point groups. If we look at, e.g., ``dipole['S_2']``:

>>> dipole['S_2']
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=object)

we see that it is a ``numpy.ndarray`` with zero for all entries. This makes sense because the point group ``S_2`` contains inversion symmetry. Let's look at a more exciting point group:

>>> dipole['C_3v']
array([[[0, xyx, yyz],
        [xyx, 0, 0],
        [yzy, 0, 0]],
       [[xyx, 0, 0],
        [0, -xyx, yyz],
        [0, yzy, 0]],
       [[zyy, 0, 0],
        [0, zyy, 0],
        [0, 0, zzz]]], dtype=object)

Let's pause to discuss two things here. For one, we see that the ``dtype`` of dipole values is ``object``. This is simply because the entries of each ``dipole`` tensor are actually ``sympy.Expr`` objects. Second, notice that ``dipole['C_3v']`` has ``yyz`` and ``yzy`` as independent elements. However, we know that these should in fact be the same, as the SHG response function ``P_i = chi_ijk E_j E_k`` is symmetric in ``j <-> k``, we should have ``chi_ijk = chi_ikj``. This type of simplification is not implemented in :mod:`shgpy.tensor_definitions`, because certain use cases actually require this symmetry not be implemented. But we can just do it manually, using :func:`shgpy.core.utilities.particularize`:

>>> from shgpy import particularize
>>> particularize(dipole['C_3v'])
array([[[0, xyx, yzy],
        [xyx, 0, 0],
        [yzy, 0, 0]],
       [[xyx, 0, 0],
        [0, -xyx, yzy],
        [0, yzy, 0]],
       [[zyy, 0, 0],
        [0, zyy, 0],
        [0, 0, zzz]]], dtype=object)

In addition to ``dipole``, there are two other dictionaries defined in :mod:`shgpy.tensor_definitions`: ``surface`` and ``quadrupole``. ``surface`` is an exact duplicate of ``dipole`` except with an ``'s'`` prepended to every parameter; e.g.

>>> from shgpy.tensor_definitions import surface
>>> surface['C_3v']
array([[[0, sxyx, syyz],
        [sxyx, 0, 0],
        [syzy, 0, 0]],
       [[sxyx, 0, 0],
        [0, -sxyx, syyz],
        [0, syzy, 0]],
       [[szyy, 0, 0],
        [0, szyy, 0],
        [0, 0, szzz]]], dtype=object)

The reason that ``surface`` exists is because sometimes you want to be able to fit a particular dataset to e.g.

>>> my_tensor = dipole['C_3v']+surface['C_3']

and this is a convenient way of doing that. But by all accounts ``dipole`` is much more frequently used.

The last tensor type we haven't talked about, ``quadrupole``, is the same idea except we're talking about quadrupole SHG so the tensor is actually rank 4. Go ahead and load a quadrupole tensor into your python session to get a feel for how it looks.

By the way, there is an ambiguity involving the direction of relevant high-symmetry axes in a given point group compared to the ``x``, ``y``, and ``z`` axes implicitly defined here. Except where otherwise noted, the convention in these definitions is to follow that of Boyd's textbook, "Nonlinear Optics." The user is encouraged to consult this textbook for further information (author's note: if there's need, I would be happy to make these definitions more explicit in the documentation, I just haven't had time. See :doc:`how to contribute <../contribute>`).

When in doubt, you can always test that the tensor you're using has the right symmetries by using :func:`shgpy.core.utilities.transform` (see the next section for more details).



Manipulating tensors
--------------------

So far we've learned how to load predefined tensors into ShgPy. But sometimes we want to use a tensor not exactly how it's written in :mod:`shgpy.tensor_definitions`, but perhaps rotated by 90 degrees or inverted. In this section, we explore the basic means provided in ShgPy for doing just that.

The most relevant function for transforming SHG tensors is :func:`shgpy.core.utilities.transform`. Let's see how this function works.

>>> from shgpy import transform
>>> import numpy as np
>>> t1 = dipole['C_3v']
>>> i = -np.identity(3, dtype=int)
>>> transform(t1, i)
array([[[0, -xyx, -yyz],
        [-xyx, 0, 0],
        [-yzy, 0, 0]],
       [[-xyx, 0, 0],
        [0, xyx, -yyz],
        [0, -yzy, 0]],
       [[-zyy, 0, 0],
        [0, -zyy, 0],
        [0, 0, -zzz]]], dtype=object)

As expected. As another example, let's transform our tensor by 3-fold rotation about the z-axis:

>>> import sympy
>>> from sghpy import rotation_matrix3symb
>>> R = rotation_matrix3symb(np.array([0, 0, 1]), 2*sympy.pi/3)
>>> transform(t1, R)
array([[[0, xyx, yyz],
        [xyx, 0, 0],
        [yzy, 0, 0]],
       [[xyx, 0, 0],
        [0, -xyx, yyz],
        [0, yzy, 0]],
       [[zyy, 0, 0],
        [0, zyy, 0],
        [0, 0, zzz]]], dtype=object)

That's good, our tensor is actually invariant under 3-fold rotation as advertised.

Before we end this tutorial, there's one more important issue we need to discuss. When you initialize a ``Symbol`` in ``sympy`` (as in ``x = sympy.symbols('x')``), there are no assumptions on that symbol except that it is commutative. In particular, the symbol is allowed to be complex. However, in ``shgpy`` it's much easier if we know exactly whether the symbol is real or imaginary. For this reason, **shgpy only accepts tensors for which all symbols are fully real**. To make sure that ``shgpy`` knows about this assumption, use

>>> t1_real = shgpy.make_tensor_real(t1)

Inspecting the elements of ``t1_real`` using ``sympy.Symbol.assumptions0`` shows us that ``make_tensor_real`` has the reality of the symbols baked in explicitly.

Of course this assumption isn't quite realistic -- for real materials, the susceptibility elements can take on any complex value, not just fully real. In those cases, we can simply decompose each symbol into its real and imaginary parts -- both of which are fully real numbers. The easy way to do this is to use ``shgpy.make_tensor_complex``:

>>> shgpy.make_tensor_complex(t1)
array([[[0, I*imag_xyx + real_xyx, I*imag_yyz + real_yyz],
    [I*imag_xyx + real_xyx, 0, 0],
    [I*imag_yzy + real_yzy, 0, 0]],
   [[I*imag_xyx + real_xyx, 0, 0],
    [0, -I*imag_xyx - real_xyx, I*imag_yyz + real_yyz],
    [0, I*imag_yzy + real_yzy, 0]],
   [[I*imag_zyy + real_zyy, 0, 0],
    [0, I*imag_zyy + real_zyy, 0],
    [0, 0, I*imag_zzz + real_zzz]]], dtype=object)

Using ``sympy.Symbol.assumptions0`` you can again inspect ``real_...`` and ``imag_...`` to prove that they are explicitly real numbers. Now your tensor is safe to start trying to fit data, as described in the next section.

This ends our tutorial on tensors in ShgPy, but feel free to peruse through the relevant :doc:`documentation <../modules>` for more info before moving on to the :doc:`next tutorial <fitting_tutorial>`.
