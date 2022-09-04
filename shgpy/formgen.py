"""Module for (non-Fourier) formula derivation

This module provides a number of routines for calculating the SHG 
response of a given material (encoded by its susceptbility tensor,
see :mod:`~shgpy.tensor_definitions` ). These routines are much
more straightforward than those found in :mod:`~shgpy.fformgen`,
but are only suited for the most simple problems because the
conversion from formulas to Fourier formulas (which is all but
necessary for efficient fitting functionality) is typically quite
slow.

"""
import inspect
from warnings import warn

import numpy as np
import sympy as sp
from .core import (
    FormContainer,
    tensor_contract,
    tensor_product,
)
from . import shg_symbols as S
from .core.utilities import _assert_real_params


# TODO
# Make all the functions here use gen_P_...


def _expand_ndarray(a, *args, **kwargs):
    return _sympy_modify_ndarray(a, sp.expand, args, kwargs)


def _expand_trig_ndarray(a, *args, **kwargs):
    return _sympy_modify_ndarray(a, sp.expand_trig, args, kwargs)


def _sympy_modify_ndarray(a, func, args, kwargs):
    ans = np.zeros(a.shape, dtype=object).flatten()
    for i, e in enumerate(a.flatten()):
        ans[i] = func(e, *args, **kwargs)
    return ans.reshape(a.shape)


def gen_levi_civita():
    ans = np.zeros((3, 3, 3))
    ans[0][1][2] = 1
    ans[1][2][0] = 1
    ans[2][0][1] = 1
    ans[1][0][2] = -1
    ans[0][2][1] = -1
    ans[2][1][0] = -1
    return ans


def make_form_from_P_and_Q(Pp, Ps, Qp, Qs):
    """Given a dipole and quadrupole moment, return the SHG signal.

    Parameters
    ----------
    Pp : sympy.Expr
        2\\omega dipole component if the input is P polarized
    Ps : sympy.Expr
        2\\omega dipole component if the input is S polarized
    Qp : sympy.Expr
        2\\omega quadrupole component if the input is P polarized
    Qs : sympy.Expr
        2\\omega quadrupole component if the input is S polarized

    Returns
    -------
    form : FormContainer

    """
    warn(
        f'Warning: {inspect.stack()[0][3]} will be deprecated by next release. Use formgen() instead.',
        category=FutureWarning,
    )

    _assert_real_params(Pp)
    _assert_real_params(Ps)
    _assert_real_params(Qp)
    _assert_real_params(Qs)

    Sp = Pp + sp.I*Qp
    Ss = Ps + sp.I*Qs

    PP = Sp[0]*sp.conjugate(Sp[0])+Sp[2]*sp.conjugate(Sp[2])
    PS = Sp[1]*sp.conjugate(Sp[1])
    SP = Ss[0]*sp.conjugate(Ss[0])+Ss[2]*sp.conjugate(Ss[2])
    SS = Ss[1]*sp.conjugate(Ss[1])

    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def gen_P_just_dipole(t1, theta):
    """Generate P formula assuming dipole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG susceptibility tensor; see :mod:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    Pp : array_like of sympy.Expr
        The 2\\omega polarization assuming P-polarized input
    Ps : array_like of sympy.Expr
        The 2\\omega polarization assuming S-polarized input

    """
    warn(
        f'Warning: {inspect.stack()[0][3]} will be deprecated by next release. Use gen_S() instead.',
        category=FutureWarning,
    )

    _assert_real_params(t1)
    c = sp.cos(theta)
    s = sp.sin(theta)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array(
        [
            [sp.cos(S.phi), -sp.sin(S.phi), 0],
            [sp.sin(S.phi), sp.cos(S.phi), 0],
            [0, 0, 1],
        ],
    )
    rotated_tensor = tensor_contract(
        tensor_product(R, R, R, t1),
        [[1, 6], [3, 7], [5, 8]],
    )
    Ps = tensor_contract(
        tensor_product(rotated_tensor, Fs, Fs),
        [[1, 3], [2, 4]],
    )
    Pp = tensor_contract(tensor_product(
        rotated_tensor, Fp, Fp),
        [[1, 3], [2, 4]],
    )
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout

    return Pp, Ps, 0, 0


def gen_P_dipole_quadrupole(t1, t2, theta):
    """Generate P formula assuming dipole+quadrupole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG dipole susceptibility tensor; see
        :mod:`~shgpy.tensor_definitions`.
    t2 : ndarray of sympy.Expr
        SHG quadrupole susceptibility tensor;
        see :mod:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    Pp_eff : array_like of sympy.Expr
        The 2\\omega effective polarization assuming P-polarized input
    Ps_eff : array_like of sympy.Expr
        The 2\\omega effective polarization assuming S-polarized input
    
    """
    warn(
        f'Warning: {inspect.stack()[0][3]} will be deprecated by next release. Use gen_S() instead.',
        category=FutureWarning,
    )
    _assert_real_params(t1)
    _assert_real_params(t2)

    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array(
        [
            [sp.cos(S.phi), -sp.sin(S.phi), 0],
            [sp.sin(S.phi), sp.cos(S.phi), 0],
            [0, 0, 1],
        ],
    )
    rotated_tensor = tensor_contract(
        tensor_product(R, R, R, t1),
        [[1, 6], [3, 7], [5, 8]],
    )
    rotated_qtensor = tensor_contract(
        tensor_product(R, R, R, R, t2),
        [[1, 8], [3, 9], [5, 10], [7, 11]],
    )
    Ps = tensor_contract(
        tensor_product(rotated_tensor, Fs, Fs),
        [[1, 3], [2, 4]],
    )
    Pp = tensor_contract(
        tensor_product(rotated_tensor, Fp, Fp),
        [[1, 3], [2, 4]],
    )
    Qs = tensor_contract(
        tensor_product(rotated_qtensor, kin, Fs, Fs),
        [[1, 4], [2, 5], [3, 6]],
    )
    Qp = tensor_contract(
        tensor_product(rotated_qtensor, kin, Fp, Fp),
        [[1, 4], [2, 5], [3, 6]],
    )
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    Qs -= np.dot(kout, Qs)*kout
    Qp -= np.dot(kout, Qp)*kout
    return Pp, Ps, Qp, Qs


def gen_S(theta, t_eee=None, t_mee=None, t_qee=None):
    if t_eee is None and t_mee is None and t_qee is None:
        raise ValueError('One of t_eee, t_mee, and t_qee must be non None.')

    def _ex(a):
        return _expand_ndarray(_expand_trig_ndarray(a))

    Sp = 0
    Ss = 0
    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array(
        [
            [sp.cos(S.phi), -sp.sin(S.phi), 0],
            [sp.sin(S.phi), sp.cos(S.phi), 0],
            [0, 0, 1],
        ],
    )
    z = gen_levi_civita()

    if t_eee is not None:
        _assert_real_params(t_eee)
        rt_eee = tensor_contract(
            tensor_product(R, R, R, t_eee),
            [[1, 6], [3, 7], [5, 8]],
        )
        rt_eee = _ex(rt_eee)
        Ps = tensor_contract(
            tensor_product(rt_eee, Fs, Fs),
            [[1, 3], [2, 4]],
        )
        Pp = tensor_contract(
            tensor_product(rt_eee, Fp, Fp),
            [[1, 3], [2, 4]],
        )
        Ss += _ex(Ps)
        Sp += _ex(Pp)
    if t_mee is not None:
        _assert_real_params(t_mee)
        rt_mee = tensor_contract(
            tensor_product(R, R, R, t_mee),
            [[1, 6], [3, 7], [5, 8]],
        )
        rt_mee = _ex(rt_mee)
        Ms = tensor_contract(
            tensor_product(z, kin, rt_mee, Fs, Fs),
            [[1, 3], [2, 4], [5, 7], [6, 8]],
        )
        Mp = tensor_contract(
            tensor_product(z, kin, rt_mee, Fp, Fp),
            [[1, 3], [2, 4], [5, 7], [6, 8]],
        )
        Ss += _ex(Ms)
        Sp += _ex(Mp)
    if t_qee is not None:
        _assert_real_params(t_qee)
        rt_qee = tensor_contract(
            tensor_product(R, R, R, t_qee),
            [[1, 6], [3, 7], [5, 8]],
        )
        rt_qee = _ex(rt_qee)
        Qs = tensor_contract(
            tensor_product(rt_qee, kin, Fs, Fs),
            [[1, 4], [2, 5], [3, 6]],
        )
        Qp = tensor_contract(
            tensor_product(rt_qee, kin, Fp, Fp),
            [[1, 4], [2, 5], [3, 6]],
        )
        Ss += _ex(sp.I*Qs)
        Sp += _ex(sp.I*Qp)

    Ss -= np.dot(kout, Ss)*kout
    Sp -= np.dot(kout, Sp)*kout

    return _ex(Sp), _ex(Ss)


def formgen(theta, t_eee=None, t_mee=None, t_qee=None):
    """Generate generic SHG formula for any multipole.

    Parameters
    ----------
    theta : float
        Angle of incidence.
    t_eee : ndarray of sympy.Expr, optional
        The electric dipole tensor. Pass `None` to neglect electric
        dipole. Default is `None`. Must pass at least one of `t_eee`,
        `t_mee`, or `t_qee`.
    t_mee : ndarray of sympy.Expr, optional
        The magnetic dipole tensor. Pass `None` to neglect magnetic
        dipole. Default is `None`. Must pass at least one of `t_eee`,
        `t_mee`, or `t_qee`.
    t_qee : ndarray of sympy.Expr, optional
        The electric quadrupole tensor. Pass `None` to neglect electric
        quadrupole. Default is `None`. Must pass at least one of `t_eee`,
        `t_mee`, or `t_qee`.
    """

    Sp, Ss = gen_S(theta, t_eee, t_mee, t_qee)

    def _ex(e):
        return sp.expand(sp.expand_trig(e))

    PP = _ex(Sp[0]*sp.conjugate(Sp[0]))+_ex(Sp[2]*sp.conjugate(Sp[2]))
    PS = _ex(Sp[1]*sp.conjugate(Sp[1]))
    SP = _ex(Ss[0]*sp.conjugate(Ss[0]))+_ex(Ss[2]*sp.conjugate(Ss[2]))
    SS = _ex(Ss[1]*sp.conjugate(Ss[1]))

    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def formgen_just_dipole(t1, theta):
    """Generate formula assuming dipole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG susceptibility tensor; see :mod:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`.

    Notes
    -----
    This routine differs from :func:`~shgpy.formgen.formgen_just_dipole_real` 
    only in the sense that the computed intensity function is computed by
    computing the modulus-squared (rather than ``**2``) of the polarization.
    For this reason, it is usually suggested to explicitly substitute 
    ``x -> real_x+1j*imag_x`` for each `x` in `t1`. See
    :func:`~shgpy.core.utilities.make_tensor_complex` and the tutorial.

    """
    warn(
        f'Warning: {inspect.stack()[0][3]} will be deprecated by next release. Use formgen() instead.',
        category=FutureWarning,
    )
    _assert_real_params(t1)

    c = sp.cos(theta)
    s = sp.sin(theta)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array(
        [
            [sp.cos(S.phi), -sp.sin(S.phi), 0],
            [sp.sin(S.phi), sp.cos(S.phi), 0],
            [0, 0, 1],
        ],
    )
    rotated_tensor = tensor_contract(
        tensor_product(R, R, R, t1),
        [[1, 6], [3, 7], [5, 8]],
    )
    Ps = tensor_contract(
        tensor_product(rotated_tensor, Fs, Fs),
        [[1, 3], [2, 4]],
    )
    Pp = tensor_contract(
        tensor_product(rotated_tensor, Fp, Fp),
        [[1, 3], [2, 4]],
    )
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    PP = Pp[0]*sp.conjugate(Pp[0])+Pp[2]*sp.conjugate(Pp[2])
    PS = Pp[1]*sp.conjugate(Pp[1])
    SP = Ps[0]*sp.conjugate(Ps[0])+Ps[2]*sp.conjugate(Ps[2])
    SS = Ps[1]*sp.conjugate(Ps[1])
    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def formgen_dipole_quadrupole(t1, t2, theta):
    """Generate formula assuming dipole+quadrupole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG dipole susceptibility tensor; see
        :mod:`~shgpy.tensor_definitions`.
    t2 : ndarray of sympy.Expr
        SHG quadrupole susceptibility tensor; see
        :mod:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`.
    
    """
    warn(
        f'Warning: {inspect.stack()[0][3]} will be deprecated by next release. Use formgen() instead.',
        category=FutureWarning,
    )
    _assert_real_params(t1)
    _assert_real_params(t2)

    theta = sp.sympify(theta)
    if not theta.is_real:
        raise ValueError('theta must be a real variable (did you forget'
                         ' real=True in sympy.symbols?)')
    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array(
        [
            [sp.cos(S.phi), -sp.sin(S.phi), 0],
            [sp.sin(S.phi), sp.cos(S.phi), 0],
            [0, 0, 1]
        ],
    )
    rotated_tensor = tensor_contract(
        tensor_product(R, R, R, t1),
        [[1, 6], [3, 7], [5, 8]],
    )
    rotated_qtensor = tensor_contract(
        tensor_product(R, R, R, R, t2),
        [[1, 8], [3, 9], [5, 10], [7, 11]],
    )
    Ps = tensor_contract(
        tensor_product(rotated_tensor, Fs, Fs),
        [[1, 3], [2, 4]],
    )
    Pp = tensor_contract(
        tensor_product(rotated_tensor, Fp, Fp),
        [[1, 3], [2, 4]],
    )
    Qs = tensor_contract(
        tensor_product(rotated_qtensor, kin, Fs, Fs),
        [[1, 4], [2, 5], [3, 6]],
    )
    Qp = tensor_contract(
        tensor_product(rotated_qtensor, kin, Fp, Fp),
        [[1, 4], [2, 5], [3, 6]],
    )
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    Qs -= np.dot(kout, Qs)*kout
    Qp -= np.dot(kout, Qp)*kout
    Sp = Pp+1j*Qp
    Ss = Ps+1j*Qs
    PP = Sp[0]*sp.conjugate(Sp[0])+Sp[2]*sp.conjugate(Sp[2])
    PS = Sp[1]*sp.conjugate(Sp[1])
    SP = Ss[0]*sp.conjugate(Ss[0])+Ss[2]*sp.conjugate(Ss[2])
    SS = Ss[1]*sp.conjugate(Ss[1])
    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})
