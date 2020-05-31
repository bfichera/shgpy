import numpy as np
import sympy as sp
from copy import deepcopy
from . import shg_symbols as S
from .core import n2i


def apply_phase_shift(fexpr, psi, M=16):
    """f(phi) -> f(phi + psi) <=> rf(m) -> exp(i*m*psi)rf(m)"""
    ans = np.zeros(shape=fexpr.shape, dtype=object)
    for m in np.arange(-M, M+1):
        ans[n2i(m, M)] = fexpr[n2i(m, M)] * (sp.cos(m*psi)+1j*sp.sin(m*psi))
    return ans


def substitute_into_array(expr_array, *subs_tuples):
    ans = np.zeros(shape=expr_array.shape, dtype=object).flatten()
    temp = expr_array.flatten()
    for i in range(len(temp)):
        try:
            ans[i] = temp[i].subs(subs_tuples)
        except AttributeError:
            ans[i] = temp[i]
    return ans.reshape(expr_array.shape)


def fform_subs(fform, subs_array, M=16):
    new_fform = deepcopy(fform)
    new_fform.subs(subs_array)
    return new_fform
