Changelog
=========

v0.7.8
------
- Added the ability to apply Kleinman symmetry to a particular SHG tensor (i.e. enforce full permutation symmetry of the susceptibility indices). Use :func:`shgpy.particularize` with the option `permute_all_indices=True`

v0.7.6
------
- Added xlabel and ylabel axes to :func:`shgpy.plotter.easy_plot`

v0.7.5
------
- Fixed a number of tensors which were incorrectly defined due to typos in Boyd. The affected tensors were
    - `D_6` (dipole)
    - `C_4` (quadrupole)
    - `C_4h` (quadrupole)
    - `S_4` (quadrupole)

v0.7.0
------
- Officially transitioned to supporting only tensors for which all the involved symbols are purely real, as defined by ``sympy`` assumptions. This got rid of a lot of redundancy in function definitions, such as ``shgpy.fformgen.generate_contracted_fourier_transforms`` versus ``shgpy.fformgen.generate_contracted_fourier_transforms_complex``, ``shgpy.formgen.formgen_dipole_quadrupole_real`` and ``shgpy.formgen.formgen_dipole_quadrupole_complex``, ect. In all cases, these functions have been replaced by a single function, e.g. ``shgpy.formgen.formgen_dipole_quadrupole``, and you will receive a ``NotImplementedError`` if you try to use any of the replaced definitions.

- To aid in explicitly defining the reality of SHG tensors, added ``shgpy.make_tensor_real`` to complement ``shgpy.make_tensor_complex``.

- Transitioned to compiling cost functions at runtime by generating C code with ``sympy.ulities.codegen``. This is a workaround to the fact that complicated ``sympy.lambdify`` functions are very slow to evaluate.

- Added the ability to generate a cost function independently with :func:`shgpy.fformfit.gen_cost_func` and use it in one of the fitting routines by the ``load_cost_func_filename`` argument.

- Added ``shgpy.fformfit.dual_annealing_fit`` and ``shgpy.fformfit.dual_annealing_fit_with_bounds``.


v0.6.1
------
- Added the ability to optionally send arguments to the ``scipy.optimize.basinhopping`` function. This is useful e.g. for debugging -- use

>>> basinhopping_kwargs = {'disp':True}

to send ``disp=True`` to ``scipy.optimize.basinhopping``, which initializes verbose output.

v0.5.1
------
- Fixed a bug related to the change in v0.5.0 in which :func:`shgpy.load_fform` wasn't compatible with the new pickling scheme.

v0.5.0
------
- In ``.p`` file handling, switched from pickling pure ``sympy`` expressions in :mod:`shgpy.fformgen` to pickling string represetations of those expressions generated using ``sympy.srepr``. This is a workaround to a ``sympy`` / ``pickle`` bug in which unpickling ``sympy`` expressions will cause ``sympy`` to conflate objects like ``sympy.Symbol('x')`` with ``sympy.Symbol('x', real=True)``.

- To use v0.5.0, you will have to remake your ``.p`` files.
