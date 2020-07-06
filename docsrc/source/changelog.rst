Changelog
=========

v0.6.1
------
Added the ability to optionally send arguments to the ``scipy.optimize.basinhopping`` function. This is useful e.g. for debugging -- use

>>> basinhopping_kwargs = {'disp':True}

to send ``disp=True`` to ``scipy.optimize.basinhopping``, which initializes verbose output.

v0.5.1
------
Fixed a bug related to the change in v0.5.0 in which :func:`shgpy.load_fform` wasn't compatible with the new pickling scheme.

v0.5.0
------
In ``.p`` file handling, switched from pickling pure ``sympy`` expressions in :mod:`shgpy.fformgen` to pickling string represetations of those expressions generated using ``sympy.srepr``. This is a workaround to a ``sympy`` / ``pickle`` bug in which unpickling ``sympy`` expressions will cause ``sympy`` to conflate objects like ``sympy.Symbol('x')`` with ``sympy.Symbol('x', real=True)``.

To use v0.5.0, you will have to remake your ``.p`` files.