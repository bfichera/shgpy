import numpy as np
import sympy as sp
from .core import (
    FormContainer,
    tensor_contract,
    tensor_product,
)
from . import shg_symbols as S


def formgen_just_dipole_complex(t1, theta):
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
