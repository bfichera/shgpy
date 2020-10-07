"""This module provides a collection of susceptibility tensors (both
dipole and quadrupole) which have each been simplified using the
symmetry constraints implied by different crystallographic point
groups. The tensors are contained in dictionaries of numpy `ndarray` s,
where the keys of the dictionaries are strings containing the name
of the relevant point group. For example, to get the dipole tensor
implied by the crystallographic point group C_4v, one would use

>>> import shgpy.tensor_definitions as td
>>> t1 = td.dipole['C_4v']

In addition to the `dipole` dictionary, this module also provides
a `quadrupole` dictionary with the rank-4 susceptibility tensors
needed to compute quadrupole SHG, as well as a special `surface`
dictionary (which is a duplicate of `dipole` but with the variables
renamed e.g. from `xxx` -> `sxxx`. This is useful when analyzing
SHG signal involving both bulk and surface dipole sources.

All of the variables defined in this module are sympy.Symbol objects
and are defined in the :mod:`~shgpy.shg_symbols` module.

The crystallographic point groups which are defined in this module are


- ``'S_2'``
- ``'C_2h'``
- ``'D_2h'``
- ``'C_4h'``
- ``'D_4h'``
- ``'T_h'``
- ``'O_h'``
- ``'S_6'``
- ``'D_3d'``
- ``'C_6h'``
- ``'D_6h'``
- ``'C_2'``
- ``'C_1h'``
- ``'D_2'``
- ``'C_2v'``
- ``'C_4'``
- ``'S_4'``
- ``'D_4'``
- ``'C_4v'``
- ``'D_2d'``
- ``'O'``
- ``'T_d'``
- ``'T'``
- ``'D_3'``
- ``'C_3'``
- ``'C_3v'``
- ``'C_6'``
- ``'C_3h'``
- ``'D_6'``
- ``'C_6v'``
- ``'D_3h'``
- ``'C_1'``

In `quadrupole`, there is an additional key ``'Isotropic'``.

Please note that an additional constraint on typical SHG tensors which
is not captured by the definitions in this module is that, since the
SHG response function is symmetric in exchange of the electric field
vectors, SHG tensors should be symmetric in their last two indices.
This constraint is not captured in these definitions because one could
imagine a broader use case in which this symmetry is not necessarily
valid. In scripts, one should use :func:`~shgpy.core.utilities.particularize`
to implement this constraint.

"""
import numpy as np
from .shg_symbols import *

dipole = {}
dipole['S_2'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_2h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_2h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_4h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_4h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['T_h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['O_h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['S_6'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_3d'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_6h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_6h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_2'] = np.array([[[0, xxy, 0], [xyx, 0, xyz], [0, xzy, 0]], [[yxx, 0, yxz], [0, yyy, 0], [yzx, 0, yzz]], [[0, zxy, 0], [zyx, 0, zyz], [0, zzy, 0]]], dtype=object)
dipole['C_1h'] = np.array([[[xxx, 0, xxz], [0, xyy, 0], [xzx, 0, xzz]], [[0, yxy, 0], [yyx, 0, yyz], [0, yzy, 0]], [[zxx, 0, zxz], [0, zyy, 0], [zzx, 0, zzz]]], dtype=object)
dipole['D_2'] = np.array([[[0, 0, 0], [0, 0, xyz], [0, xzy, 0]], [[0, 0, yxz], [0, 0, 0], [yzx, 0, 0]], [[0, zxy, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_2v'] = np.array([[[0, 0, xxz], [0, 0, 0], [xzx, 0, 0]], [[0, 0, 0], [0, 0, yyz], [0, yzy, 0]], [[zxx, 0, 0], [0, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['C_4'] = np.array([[[0, 0, yyz], [0, 0, -yxz], [yzy, -yzx, 0]], [[0, 0, yxz], [0, 0, yyz], [yzx, yzy, 0]], [[zyy, -zyx, 0], [zyx, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['S_4'] = np.array([[[0, 0, -yyz], [0, 0, yxz], [-yzy, yzx, 0]], [[0, 0, yxz], [0, 0, yyz], [yzx, yzy, 0]], [[-zyy, zyx, 0], [zyx, zyy, 0], [0, 0, 0]]], dtype=object)
dipole['D_4'] = np.array([[[0, 0, 0], [0, 0, -yxz], [0, -yzx, 0]], [[0, 0, yxz], [0, 0, 0], [yzx, 0, 0]], [[0, -zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_4v'] = np.array([[[0, 0, yyz], [0, 0, 0], [yzy, 0, 0]], [[0, 0, 0], [0, 0, yyz], [0, yzy, 0]], [[zyy, 0, 0], [0, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['D_2d'] = np.array([[[0, 0, 0], [0, 0, yxz], [0, yzx, 0]], [[0, 0, yxz], [0, 0, 0], [yzx, 0, 0]], [[0, zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['O'] = np.array([[[0, 0, 0], [0, 0, -zyx], [0, zyx, 0]], [[0, 0, zyx], [0, 0, 0], [-zyx, 0, 0]], [[0, -zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['T_d'] = np.array([[[0, 0, 0], [0, 0, zyx], [0, zyx, 0]], [[0, 0, zyx], [0, 0, 0], [zyx, 0, 0]], [[0, zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['T'] = np.array([[[0, 0, 0], [0, 0, zxy], [0, zyx, 0]], [[0, 0, zyx], [0, 0, 0], [zxy, 0, 0]], [[0, zxy, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_3'] = np.array([[[-yxy, 0, 0], [0, yxy, -yxz], [0, -yzx, 0]], [[0, yxy, yxz], [yxy, 0, 0], [yzx, 0, 0]], [[0, -zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_3'] = np.array([[[xxx, -yyy, yyz], [-yyy, -xxx, -yxz], [yzy, -yzx, 0]], [[-yyy, -xxx, yxz], [-xxx, yyy, yyz], [yzx, yzy, 0]], [[zyy, -zyx, 0], [zyx, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['C_3v'] = np.array([[[0, xyx, yyz], [xyx, 0, 0], [yzy, 0, 0]], [[xyx, 0, 0], [0, -xyx, yyz], [0, yzy, 0]], [[zyy, 0, 0], [0, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['C_6'] = np.array([[[0, 0, yyz], [0, 0, -yxz], [yzy, -yzx, 0]], [[0, 0, yxz], [0, 0, yyz], [yzx, yzy, 0]], [[zyy, -zyx, 0], [zyx, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['C_3h'] = np.array([[[-yyx, xxy, 0], [xxy, yyx, 0], [0, 0, 0]], [[xxy, yyx, 0], [yyx, -xxy, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_6v'] = np.array([[[0, 0, yyz], [0, 0, 0], [yzy, 0, 0]], [[0, 0, 0], [0, 0, yyz], [0, yzy, 0]], [[zyy, 0, 0], [0, zyy, 0], [0, 0, zzz]]], dtype=object)
dipole['D_6'] = np.array([[[0, 0, 0], [0, 0, -yxz], [0, -yzx, 0]], [[0, 0, yxz], [0, 0, 0], [yzx, 0, 0]], [[0, -zyx, 0], [zyx, 0, 0], [0, 0, 0]]], dtype=object)
dipole['D_3h'] = np.array([[[0, xyx, 0], [xyx, 0, 0], [0, 0, 0]], [[xyx, 0, 0], [0, -xyx, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
dipole['C_1'] = np.array([[[xxx, xxy, xxz], [xyx, xyy, xyz], [xzx, xzy, xzz]], [[yxx, yxy, yxz], [yyx, yyy, yyz], [yzx, yzy, yzz]], [[zxx, zxy, zxz], [zyx, zyy, zyz], [zzx, zzy, zzz]]], dtype=object)
for k,v in dipole.items():
    new_dipole = np.zeros(shape=v.shape, dtype=object).flatten()
    for i,g in enumerate(dipole[k].flatten()):
        new_dipole[i] = sp.sympify(g)
    dipole[k] = new_dipole.reshape(v.shape)

surface = {}
surface['S_2'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_2h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_2h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_4h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_4h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['T_h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['O_h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['S_6'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_3d'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_6h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_6h'] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_2'] = np.array([[[0, sxxy, 0], [sxyx, 0, sxyz], [0, sxzy, 0]], [[syxx, 0, syxz], [0, syyy, 0], [syzx, 0, syzz]], [[0, szxy, 0], [szyx, 0, szyz], [0, szzy, 0]]], dtype=object)
surface['C_1h'] = np.array([[[sxxx, 0, sxxz], [0, sxyy, 0], [sxzx, 0, sxzz]], [[0, syxy, 0], [syyx, 0, syyz], [0, syzy, 0]], [[szxx, 0, szxz], [0, szyy, 0], [szzx, 0, szzz]]], dtype=object)
surface['D_2'] = np.array([[[0, 0, 0], [0, 0, sxyz], [0, sxzy, 0]], [[0, 0, syxz], [0, 0, 0], [syzx, 0, 0]], [[0, szxy, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_2v'] = np.array([[[0, 0, sxxz], [0, 0, 0], [sxzx, 0, 0]], [[0, 0, 0], [0, 0, syyz], [0, syzy, 0]], [[szxx, 0, 0], [0, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['C_4'] = np.array([[[0, 0, syyz], [0, 0, -syxz], [syzy, -syzx, 0]], [[0, 0, syxz], [0, 0, syyz], [syzx, syzy, 0]], [[szyy, -szyx, 0], [szyx, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['S_4'] = np.array([[[0, 0, -syyz], [0, 0, syxz], [-syzy, syzx, 0]], [[0, 0, syxz], [0, 0, syyz], [syzx, syzy, 0]], [[-szyy, szyx, 0], [szyx, szyy, 0], [0, 0, 0]]], dtype=object)
surface['D_4'] = np.array([[[0, 0, 0], [0, 0, -syxz], [0, -syzx, 0]], [[0, 0, syxz], [0, 0, 0], [syzx, 0, 0]], [[0, -szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_4v'] = np.array([[[0, 0, syyz], [0, 0, 0], [syzy, 0, 0]], [[0, 0, 0], [0, 0, syyz], [0, syzy, 0]], [[szyy, 0, 0], [0, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['D_2d'] = np.array([[[0, 0, 0], [0, 0, syxz], [0, syzx, 0]], [[0, 0, syxz], [0, 0, 0], [syzx, 0, 0]], [[0, szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['O'] = np.array([[[0, 0, 0], [0, 0, -szyx], [0, szyx, 0]], [[0, 0, szyx], [0, 0, 0], [-szyx, 0, 0]], [[0, -szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['T_d'] = np.array([[[0, 0, 0], [0, 0, szyx], [0, szyx, 0]], [[0, 0, szyx], [0, 0, 0], [szyx, 0, 0]], [[0, szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['T'] = np.array([[[0, 0, 0], [0, 0, szxy], [0, szyx, 0]], [[0, 0, szyx], [0, 0, 0], [szxy, 0, 0]], [[0, szxy, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_3'] = np.array([[[-syxy, 0, 0], [0, syxy, -syxz], [0, -syzx, 0]], [[0, syxy, syxz], [syxy, 0, 0], [syzx, 0, 0]], [[0, -szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_3'] = np.array([[[sxxx, -syyy, syyz], [-syyy, -sxxx, -syxz], [syzy, -syzx, 0]], [[-syyy, -sxxx, syxz], [-sxxx, syyy, syyz], [syzx, syzy, 0]], [[szyy, -szyx, 0], [szyx, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['C_3v'] = np.array([[[0, sxyx, syyz], [sxyx, 0, 0], [syzy, 0, 0]], [[sxyx, 0, 0], [0, -sxyx, syyz], [0, syzy, 0]], [[szyy, 0, 0], [0, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['C_6'] = np.array([[[0, 0, syyz], [0, 0, -syxz], [syzy, -syzx, 0]], [[0, 0, syxz], [0, 0, syyz], [syzx, syzy, 0]], [[szyy, -szyx, 0], [szyx, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['C_3h'] = np.array([[[-syyx, sxxy, 0], [sxxy, syyx, 0], [0, 0, 0]], [[sxxy, syyx, 0], [syyx, -sxxy, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['D_6'] = np.array([[[0, 0, 0], [0, 0, -syxz], [0, -syzx, 0]], [[0, 0, syxz], [0, 0, 0], [syzx, 0, 0]], [[0, -szyx, 0], [szyx, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_6v'] = np.array([[[0, 0, syyz], [0, 0, 0], [syzy, 0, 0]], [[0, 0, 0], [0, 0, syyz], [0, syzy, 0]], [[szyy, 0, 0], [0, szyy, 0], [0, 0, szzz]]], dtype=object)
surface['D_3h'] = np.array([[[0, sxyx, 0], [sxyx, 0, 0], [0, 0, 0]], [[sxyx, 0, 0], [0, -sxyx, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=object)
surface['C_1'] = np.array([[[sxxx, sxxy, sxxz], [sxyx, sxyy, sxyz], [sxzx, sxzy, sxzz]], [[syxx, syxy, syxz], [syyx, syyy, syyz], [syzx, syzy, syzz]], [[szxx, szxy, szxz], [szyx, szyy, szyz], [szzx, szzy, szzz]]], dtype=object)
for k,v in surface.items():
    new_surface = np.zeros(shape=v.shape, dtype=object).flatten()
    for i,g in enumerate(surface[k].flatten()):
        new_surface[i] = sp.sympify(g)
    surface[k] = new_surface.reshape(v.shape)

quadrupole={}
quadrupole['Isotropic'] = np.array([[[[yxxy + yxyx + yyxx, 0, 0], [0, yyxx, 0], [0, 0, yyxx]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yxxy + yxyx + yyxx, 0], [0, 0, yyxx]], [[0, 0, 0], [0, 0, yxyx], [0, yxxy, 0]]], [[[0, 0, yxxy], [0, 0, 0], [yxyx, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[yyxx, 0, 0], [0, yyxx, 0], [0, 0, yxxy + yxyx + yyxx]]]], dtype=object)
quadrupole['T'] = np.array([[[[zzzz, 0, 0], [0, xxyy, 0], [0, 0, yyxx]], [[0, xyxy, 0], [xyyx, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, zzzz, 0], [0, 0, xxyy]], [[0, 0, 0], [0, 0, xyxy], [0, xyyx, 0]]], [[[0, 0, xyyx], [0, 0, 0], [xyxy, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[xxyy, 0, 0], [0, yyxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['T_h'] = np.array([[[[zzzz, 0, 0], [0, xxyy, 0], [0, 0, yyxx]], [[0, xyxy, 0], [xyyx, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, zzzz, 0], [0, 0, xxyy]], [[0, 0, 0], [0, 0, xyxy], [0, xyyx, 0]]], [[[0, 0, xyyx], [0, 0, 0], [xyxy, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[xxyy, 0, 0], [0, yyxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['O_h'] = np.array([[[[zzzz, 0, 0], [0, yyxx, 0], [0, 0, yyxx]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, zzzz, 0], [0, 0, yyxx]], [[0, 0, 0], [0, 0, yxyx], [0, yxxy, 0]]], [[[0, 0, yxxy], [0, 0, 0], [yxyx, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[yyxx, 0, 0], [0, yyxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['O'] = np.array([[[[zzzz, 0, 0], [0, yyxx, 0], [0, 0, yyxx]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, zzzz, 0], [0, 0, yyxx]], [[0, 0, 0], [0, 0, yxyx], [0, yxxy, 0]]], [[[0, 0, yxxy], [0, 0, 0], [yxyx, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[yyxx, 0, 0], [0, yyxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['T_d'] = np.array([[[[zzzz, 0, 0], [0, yyxx, 0], [0, 0, yyxx]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, yxyx], [0, 0, 0], [yxxy, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, zzzz, 0], [0, 0, yyxx]], [[0, 0, 0], [0, 0, yxyx], [0, yxxy, 0]]], [[[0, 0, yxxy], [0, 0, 0], [yxyx, 0, 0]], [[0, 0, 0], [0, 0, yxxy], [0, yxyx, 0]], [[yyxx, 0, 0], [0, yyxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_6'] = np.array([[[[yxxy + yxyx + yyxx, -xxyx - xyxx - yxxx, 0], [xxyx, yyxx, 0], [0, 0, xxzz]], [[xyxx, yxyx, 0], [yxxy, -yxxx, 0], [0, 0, -yxzz]], [[0, 0, xzxz], [0, 0, -yzxz], [xzzx, -yzzx, 0]]], [[[yxxx, yxxy, 0], [yxyx, -xyxx, 0], [0, 0, yxzz]], [[yyxx, -xxyx, 0], [xxyx + xyxx + yxxx, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, yzxz], [0, 0, xzxz], [yzzx, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, -zyxz], [zxzx, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zxxz], [zyzx, zxzx, 0]], [[zzxx, -zzyx, 0], [zzyx, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_3h'] = np.array([[[[yxxy + yxyx + yyxx, -xxyx - xyxx - yxxx, 0], [xxyx, yyxx, 0], [0, 0, xxzz]], [[xyxx, yxyx, 0], [yxxy, -yxxx, 0], [0, 0, -yxzz]], [[0, 0, xzxz], [0, 0, -yzxz], [xzzx, -yzzx, 0]]], [[[yxxx, yxxy, 0], [yxyx, -xyxx, 0], [0, 0, yxzz]], [[yyxx, -xxyx, 0], [xxyx + xyxx + yxxx, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, yzxz], [0, 0, xzxz], [yzzx, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, -zyxz], [zxzx, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zxxz], [zyzx, zxzx, 0]], [[zzxx, -zzyx, 0], [zzyx, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_6h'] = np.array([[[[yxxy + yxyx + yyxx, -xxyx - xyxx - yxxx, 0], [xxyx, yyxx, 0], [0, 0, xxzz]], [[xyxx, yxyx, 0], [yxxy, -yxxx, 0], [0, 0, -yxzz]], [[0, 0, xzxz], [0, 0, -yzxz], [xzzx, -yzzx, 0]]], [[[yxxx, yxxy, 0], [yxyx, -xyxx, 0], [0, 0, yxzz]], [[yyxx, -xxyx, 0], [xxyx + xyxx + yxxx, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, yzxz], [0, 0, xzxz], [yzzx, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, -zyxz], [zxzx, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zxxz], [zyzx, zxzx, 0]], [[zzxx, -zzyx, 0], [zzyx, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_6v'] = np.array([[[[yxxy + yxyx + yyxx, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_6'] = np.array([[[[yxxy + yxyx + yyxx, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_6h'] = np.array([[[[yxxy + yxyx + yyxx, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_3h'] = np.array([[[[yxxy + yxyx + yyxx, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yxxy + yxyx + yyxx, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_3'] = np.array([[[[yxxy + yxyx + yyxx, -xxyx - xyxx - yxxx, -yyxz], [xxyx, yyxx, xxyz], [-yyzx, xxzy, xxzz]], [[xyxx, yxyx, xxyz], [yxxy, -yxxx, yyxz], [xxzy, yyzx, -yxzz]], [[-xzyy, xzxy, xzxz], [xzxy, xzyy, -yzxz], [xzzx, -yzzx, 0]]], [[[yxxx, yxxy, xxyz], [yxyx, -xyxx, yyxz], [xxzy, yyzx, yxzz]], [[yyxx, -xxyx, yyxz], [xxyx + xyxx + yxxx, yxxy + yxyx + yyxx, -xxyz], [yyzx, -xxzy, xxzz]], [[xzxy, xzyy, yzxz], [xzyy, -xzxy, xzxz], [yzzx, xzzx, 0]]], [[[-zyyx, zxxy, zxxz], [zxxy, zyyx, -zyxz], [zxzx, -zyzx, 0]], [[zxxy, zyyx, zyxz], [zyyx, -zxxy, zxxz], [zyzx, zxzx, 0]], [[zzxx, -zzyx, 0], [zzyx, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['S_6'] = np.array([[[[yxxy + yxyx + yyxx, -xxyx - xyxx - yxxx, -yyxz], [xxyx, yyxx, xxyz], [-yyzx, xxzy, xxzz]], [[xyxx, yxyx, xxyz], [yxxy, -yxxx, yyxz], [xxzy, yyzx, -yxzz]], [[-xzyy, xzxy, xzxz], [xzxy, xzyy, -yzxz], [xzzx, -yzzx, 0]]], [[[yxxx, yxxy, xxyz], [yxyx, -xyxx, yyxz], [xxzy, yyzx, yxzz]], [[yyxx, -xxyx, yyxz], [xxyx + xyxx + yxxx, yxxy + yxyx + yyxx, -xxyz], [yyzx, -xxzy, xxzz]], [[xzxy, xzyy, yzxz], [xzyy, -xzxy, xzxz], [yzzx, xzzx, 0]]], [[[-zyyx, zxxy, zxxz], [zxxy, zyyx, -zyxz], [zxzx, -zyzx, 0]], [[zxxy, zyyx, zyxz], [zyyx, -zxxy, zxxz], [zyzx, zxzx, 0]], [[zzxx, -zzyx, 0], [zzyx, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_3v'] = np.array([[[[yxxy + yxyx + yyxx, 0, -yyxz], [0, yyxx, 0], [-yyzx, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, yyxz], [0, yyzx, 0]], [[-yzyx, 0, xzxz], [0, yzyx, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, yyxz], [0, yyzx, 0]], [[yyxx, 0, yyxz], [0, yxxy + yxyx + yyxx, 0], [yyzx, 0, xxzz]], [[0, yzyx, 0], [yzyx, 0, xzxz], [0, xzzx, 0]]], [[[-zyyx, 0, zxxz], [0, zyyx, 0], [zxzx, 0, 0]], [[0, zyyx, 0], [zyyx, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_3d'] = np.array([[[[yxxy + yxyx + yyxx, 0, -yyxz], [0, yyxx, 0], [-yyzx, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, yyxz], [0, yyzx, 0]], [[-yzyx, 0, xzxz], [0, yzyx, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, yyxz], [0, yyzx, 0]], [[yyxx, 0, yyxz], [0, yxxy + yxyx + yyxx, 0], [yyzx, 0, xxzz]], [[0, yzyx, 0], [yzyx, 0, xzxz], [0, xzzx, 0]]], [[[-zyyx, 0, zxxz], [0, zyyx, 0], [zxzx, 0, 0]], [[0, zyyx, 0], [zyyx, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_3'] = np.array([[[[yxxy + yxyx + yyxx, 0, -yyxz], [0, yyxx, 0], [-yyzx, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, yyxz], [0, yyzx, 0]], [[-yzyx, 0, xzxz], [0, yzyx, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, yyxz], [0, yyzx, 0]], [[yyxx, 0, yyxz], [0, yxxy + yxyx + yyxx, 0], [yyzx, 0, xxzz]], [[0, yzyx, 0], [yzyx, 0, xzxz], [0, xzzx, 0]]], [[[-zyyx, 0, zxxz], [0, zyyx, 0], [zxzx, 0, 0]], [[0, zyyx, 0], [zyyx, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_4'] = np.array([[[[yyyy, -yyyx, 0], [-yyxy, yyxx, 0], [0, 0, yyzz]], [[-yxyy, yxyx, 0], [yxxy, xyyy, 0], [0, 0, -yxzz]], [[0, 0, yzyz], [0, 0, -yzxz], [yzzy, -yzzx, 0]]], [[[-xyyy, yxxy, 0], [yxyx, yxyy, 0], [0, 0, yxzz]], [[yyxx, yyxy, 0], [yyyx, yyyy, 0], [0, 0, yyzz]], [[0, 0, yzxz], [0, 0, yzyz], [yzzx, yzzy, 0]]], [[[0, 0, zyyz], [0, 0, -zyxz], [zyzy, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zyyz], [zyzx, zyzy, 0]], [[zzyy, -zzyx, 0], [zzyx, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_4h'] = np.array([[[[yyyy, -yyyx, 0], [-yyxy, yyxx, 0], [0, 0, yyzz]], [[-yxyy, yxyx, 0], [yxxy, xyyy, 0], [0, 0, -yxzz]], [[0, 0, yzyz], [0, 0, -yzxz], [yzzy, -yzzx, 0]]], [[[-xyyy, yxxy, 0], [yxyx, yxyy, 0], [0, 0, yxzz]], [[yyxx, yyxy, 0], [yyyx, yyyy, 0], [0, 0, yyzz]], [[0, 0, yzxz], [0, 0, yzyz], [yzzx, yzzy, 0]]], [[[0, 0, zyyz], [0, 0, -zyxz], [zyzy, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zyyz], [zyzx, zyzy, 0]], [[zzyy, -zzyx, 0], [zzyx, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['S_4'] = np.array([[[[yyyy, -yyyx, 0], [-yyxy, yyxx, 0], [0, 0, yyzz]], [[-yxyy, yxyx, 0], [yxxy, xyyy, 0], [0, 0, -yxzz]], [[0, 0, yzyz], [0, 0, -yzxz], [yzzy, -yzzx, 0]]], [[[-xyyy, yxxy, 0], [yxyx, yxyy, 0], [0, 0, yxzz]], [[yyxx, yyxy, 0], [yyyx, yyyy, 0], [0, 0, yyzz]], [[0, 0, yzxz], [0, 0, yzyz], [yzzx, yzzy, 0]]], [[[0, 0, zyyz], [0, 0, -zyxz], [zyzy, -zyzx, 0]], [[0, 0, zyxz], [0, 0, zyyz], [zyzx, zyzy, 0]], [[zzyy, -zzyx, 0], [zzyx, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_4v'] = np.array([[[[yyyy, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_4'] = np.array([[[[yyyy, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_4h'] = np.array([[[[yyyy, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_2d'] = np.array([[[[yyyy, 0, 0], [0, yyxx, 0], [0, 0, xxzz]], [[0, yxyx, 0], [yxxy, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, xxzz]], [[0, 0, 0], [0, 0, xzxz], [0, xzzx, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zxxz], [0, zxzx, 0]], [[zzxx, 0, 0], [0, zzxx, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_2'] = np.array([[[[xxxx, 0, xxxz], [0, xxyy, 0], [xxzx, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, xyyz], [0, xyzy, 0]], [[xzxx, 0, xzxz], [0, xzyy, 0], [xzzx, 0, xzzz]]], [[[0, yxxy, 0], [yxyx, 0, yxyz], [0, yxzy, 0]], [[yyxx, 0, yyxz], [0, yyyy, 0], [yyzx, 0, yyzz]], [[0, yzxy, 0], [yzyx, 0, yzyz], [0, yzzy, 0]]], [[[zxxx, 0, zxxz], [0, zxyy, 0], [zxzx, 0, zxzz]], [[0, zyxy, 0], [zyyx, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, zzxz], [0, zzyy, 0], [zzzx, 0, zzzz]]]], dtype=object)
quadrupole['C_1h'] = np.array([[[[xxxx, 0, xxxz], [0, xxyy, 0], [xxzx, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, xyyz], [0, xyzy, 0]], [[xzxx, 0, xzxz], [0, xzyy, 0], [xzzx, 0, xzzz]]], [[[0, yxxy, 0], [yxyx, 0, yxyz], [0, yxzy, 0]], [[yyxx, 0, yyxz], [0, yyyy, 0], [yyzx, 0, yyzz]], [[0, yzxy, 0], [yzyx, 0, yzyz], [0, yzzy, 0]]], [[[zxxx, 0, zxxz], [0, zxyy, 0], [zxzx, 0, zxzz]], [[0, zyxy, 0], [zyyx, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, zzxz], [0, zzyy, 0], [zzzx, 0, zzzz]]]], dtype=object)
quadrupole['C_2h'] = np.array([[[[xxxx, 0, xxxz], [0, xxyy, 0], [xxzx, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, xyyz], [0, xyzy, 0]], [[xzxx, 0, xzxz], [0, xzyy, 0], [xzzx, 0, xzzz]]], [[[0, yxxy, 0], [yxyx, 0, yxyz], [0, yxzy, 0]], [[yyxx, 0, yyxz], [0, yyyy, 0], [yyzx, 0, yyzz]], [[0, yzxy, 0], [yzyx, 0, yzyz], [0, yzzy, 0]]], [[[zxxx, 0, zxxz], [0, zxyy, 0], [zxzx, 0, zxzz]], [[0, zyxy, 0], [zyyx, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, zzxz], [0, zzyy, 0], [zzzx, 0, zzzz]]]], dtype=object)
quadrupole['C_2v'] = np.array([[[[xxxx, 0, 0], [0, xxyy, 0], [0, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, yyzz]], [[0, 0, 0], [0, 0, yzyz], [0, yzzy, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, 0], [0, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_2'] = np.array([[[[xxxx, 0, 0], [0, xxyy, 0], [0, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, yyzz]], [[0, 0, 0], [0, 0, yzyz], [0, yzzy, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, 0], [0, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['D_2h'] = np.array([[[[xxxx, 0, 0], [0, xxyy, 0], [0, 0, xxzz]], [[0, xyxy, 0], [xyyx, 0, 0], [0, 0, 0]], [[0, 0, xzxz], [0, 0, 0], [xzzx, 0, 0]]], [[[0, yxxy, 0], [yxyx, 0, 0], [0, 0, 0]], [[yyxx, 0, 0], [0, yyyy, 0], [0, 0, yyzz]], [[0, 0, 0], [0, 0, yzyz], [0, yzzy, 0]]], [[[0, 0, zxxz], [0, 0, 0], [zxzx, 0, 0]], [[0, 0, 0], [0, 0, zyyz], [0, zyzy, 0]], [[zzxx, 0, 0], [0, zzyy, 0], [0, 0, zzzz]]]], dtype=object)
quadrupole['C_1'] = np.array([[[[xxxx, xxxy, xxxz],
    [xxyx, xxyy, xxyz],
    [xxzx, xxzy, xxzz]],
    [[xyxx, xyxy, xyxz],
    [xyyx, xyyy, xyyz],
    [xyzx, xyzy, xyzz]],
    [[xzxx, xzxy, xzxz],
    [xzyx, xzyy, xzyz],
    [xzzx, xzzy, xzzz]]],
    [[[yxxx, yxxy, yxxz],
    [yxyx, yxyy, yxyz],
    [yxzx, yxzy, yxzz]],
    [[yyxx, yyxy, yyxz],
    [yyyx, yyyy, yyyz],
    [yyzx, yyzy, yyzz]],
    [[yzxx, yzxy, yzxz],
    [yzyx, yzyy, yzyz],
    [yzzx, yzzy, yzzz]]],
    [[[zxxx, zxxy, zxxz],
    [zxyx, zxyy, zxyz],
    [zxzx, zxzy, zxzz]],
    [[zyxx, zyxy, zyxz],
    [zyyx, zyyy, zyyz],
    [zyzx, zyzy, zyzz]],
    [[zzxx, zzxy, zzxz],
    [zzyx, zzyy, zzyz],
    [zzzx, zzzy, zzzz]]]], dtype=object)
for k,v in quadrupole.items():
    new_quadrupole = np.zeros(shape=v.shape, dtype=object).flatten()
    for i,g in enumerate(quadrupole[k].flatten()):
        new_quadrupole[i] = sp.sympify(g)
    quadrupole[k] = new_quadrupole.reshape(v.shape)
