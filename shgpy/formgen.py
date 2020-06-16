"""Module for (non-Fourier) formula derivation

This module provides a number of routines for calculating the SHG 
response of a given material (encoded by its susceptbility tensor,
see :class:`~shgpy.tensor_definitions` ). These routines are much
more straightforward than those found in :class:`~shgpy.fformgen`,
but are only suited for the most simple problems because the
conversion from formulas to Fourier formulas (which is all but
necessary for efficient fitting functionality) is typically quite
slow.

"""
import numpy as np
import sympy as sp
from .core import (
    FormContainer,
    tensor_contract,
    tensor_product,
)
from . import shg_symbols as S


# TODO
# Make all the functions here use gen_P_...


def make_form_from_P_and_Q(Pp, Ps, Qp, Qs, mode):
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
    mode : {'real', 'complex'}
        Whether to treat the sympy.Symbols as real or complex.

    Returns
    -------
    form : FormContainer

    """
    if mode == 'complex':

        Sp = Pp + sp.I*Qp
        Ss = Ps + sp.I*Qs

        PP = Sp[0]*sp.conjugate(Sp[0])+Sp[2]*sp.conjugate(Sp[2])
        PS = Sp[1]*sp.conjugate(Sp[1])
        SP = Ss[0]*sp.conjugate(Ss[0])+Ss[2]*sp.conjugate(Ss[2])
        SS = Ss[1]*sp.conjugate(Ss[1])

    elif mode == 'real':

        PP = Pp[0]**2+Pp[2]**2+Qp[0]**2+Qp[2]**2
        PS = Pp[1]**2+Qp[1]**2
        SP = Ps[0]**2+Ps[2]**2+Qs[0]**2+Qs[2]**2
        SS = Ps[1]**2+Qs[1]**2

    else:

        raise ValueError('mode must be either \'real\' or \'complex\'')

    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def gen_P_just_dipole(t1, theta):
    """Generate P formula assuming dipole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    Pp : array_like of sympy.Expr
        The 2\\omega polarization assuming P-polarized input
    Ps : array_like of sympy.Expr
        The 2\\omega polarization assuming S-polarized input

    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout

    return Pp, Ps, 0, 0


def gen_P_dipole_quadrupole(t1, t2, theta):
    """Generate P formula assuming dipole+quadrupole SHG.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG dipole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    t2 : ndarray of sympy.Expr
        SHG quadrupole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    Pp_eff : array_like of sympy.Expr
        The 2\\omega effective polarization assuming P-polarized input
    Ps_eff : array_like of sympy.Expr
        The 2\\omega effective polarization assuming S-polarized input
    
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    rotated_qtensor = tensor_contract(tensor_product(R, R, R, R, t2), [[1, 8], [3, 9], [5, 10], [7, 11]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Qs = tensor_contract(tensor_product(rotated_qtensor, kin, Fs, Fs), [[1, 4], [2, 5], [3, 6]])
    Qp = tensor_contract(tensor_product(rotated_qtensor, kin, Fp, Fp), [[1, 4], [2, 5], [3, 6]])
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    Qs -= np.dot(kout, Qs)*kout
    Qp -= np.dot(kout, Qp)*kout
    return Pp, Ps, Qp, Qs


def formgen_just_dipole_complex(t1, theta):
    """Generate formula assuming dipole SHG with complex coefficients.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
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
    ``x -> real_x+1j*imag_x`` for each `x` in `t1`. See :func:`~shgpy.core.utilities.make_tensor_complex`
    and the tutorial.

    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    PP = Pp[0]*sp.conjugate(Pp[0])+Pp[2]*sp.conjugate(Pp[2])
    PS = Pp[1]*sp.conjugate(Pp[1])
    SP = Ps[0]*sp.conjugate(Ps[0])+Ps[2]*sp.conjugate(Ps[2])
    SS = Ps[1]*sp.conjugate(Ps[1])
    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def formgen_just_dipole_real(t1, theta):
    """Generate formula assuming dipole SHG with real coefficients.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`.

    Notes
    -----
    This routine differs from :func:`~shgpy.formgen.formgen_just_dipole_complex` 
    only in the sense that the computed intensity function is computed by
    computing ``**2`` (rather than the modulus squared) of the polarization.
    This is usually easier as long as you're not worrying about the imaginary
    component of the susceptibility (valid away from resonance in the presence
    of time-reversal symmetry).
    

    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    PP = Pp[0]**2+Pp[2]**2
    PS = Pp[1]**2
    SP = Ps[0]**2+Ps[2]**2
    SS = Ps[1]**2
    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def formgen_dipole_quadrupole_real(t1, t2, theta):
    """Generate formula assuming dipole+quadrupole SHG with real coefficients.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG dipole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    t2 : ndarray of sympy.Expr
        SHG quadrupole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`.

    Notes
    -----
    See note in :func:`~shgpy.formgen.formgen_just_dipole_real`.
    
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    rotated_qtensor = tensor_contract(tensor_product(R, R, R, R, t2), [[1, 8], [3, 9], [5, 10], [7, 11]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Qs = tensor_contract(tensor_product(rotated_qtensor, kin, Fs, Fs), [[1, 4], [2, 5], [3, 6]])
    Qp = tensor_contract(tensor_product(rotated_qtensor, kin, Fp, Fp), [[1, 4], [2, 5], [3, 6]])
    Ps -= np.dot(kout, Ps)*kout
    Pp -= np.dot(kout, Pp)*kout
    Qs -= np.dot(kout, Qs)*kout
    Qp -= np.dot(kout, Qp)*kout
    PP = Pp[0]**2+Pp[2]**2+Qp[0]**2+Qp[2]**2
    PS = Pp[1]**2+Qp[1]**2
    SP = Ps[0]**2+Ps[2]**2+Qs[0]**2+Qs[2]**2
    SS = Ps[1]**2+Qs[1]**2
    return FormContainer({'PP':PP, 'PS':PS, 'SP':SP, 'SS':SS})


def formgen_dipole_quadrupole_complex(t1, t2, theta):
    """Generate formula assuming dipole+quadrupole SHG with complex coefficients.

    Parameters
    ----------
    t1 : ndarray of sympy.Expr
        SHG dipole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    t2 : ndarray of sympy.Expr
        SHG quadrupole susceptibility tensor; see :class:`~shgpy.tensor_definitions`.
    theta : float or sympy.Symbol
        Angle of incidence

    Returns
    -------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`.

    Notes
    -----
    See note in :func:`~shgpy.formgen.formgen_just_dipole_complex`.
    
    """
    theta = sp.sympify(theta)
    if not theta.is_real:
        raise ValueError('theta must be a real variable (did you forget real=True in sympy.symbols?)')
    c = sp.cos(theta)
    s = sp.sin(theta)
    kin = np.array([s, 0, c], dtype=object)
    kout = np.array([s, 0, -c], dtype=object)
    Fp = np.array([-c, 0, s], dtype=object)
    Fs = np.array([0, 1, 0], dtype=object)
    R = np.array([[sp.cos(S.phi), -sp.sin(S.phi), 0], [sp.sin(S.phi), sp.cos(S.phi), 0], [0, 0, 1]])
    rotated_tensor = tensor_contract(tensor_product(R, R, R, t1), [[1, 6], [3, 7], [5, 8]])
    rotated_qtensor = tensor_contract(tensor_product(R, R, R, R, t2), [[1, 8], [3, 9], [5, 10], [7, 11]])
    Ps = tensor_contract(tensor_product(rotated_tensor, Fs, Fs), [[1, 3], [2, 4]])
    Pp = tensor_contract(tensor_product(rotated_tensor, Fp, Fp), [[1, 3], [2, 4]])
    Qs = tensor_contract(tensor_product(rotated_qtensor, kin, Fs, Fs), [[1, 4], [2, 5], [3, 6]])
    Qp = tensor_contract(tensor_product(rotated_qtensor, kin, Fp, Fp), [[1, 4], [2, 5], [3, 6]])
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
