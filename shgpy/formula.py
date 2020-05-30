import numpy as np
import sympy as sp
from . import shg_symbols as S
from .core import n2i
from . import tensor as tx


def formula_from_fexpr(t, M=16):
    expr = 0
    for m in np.arange(-M, M+1):
        expr += t[n2i(m)]*(sp.cos(m*S.phi)+1j*sp.sin(m*S.phi))
    return expr


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


def extract_free_symbols_from_fform_dict(fform_dict, M=16):
    free_symbols = tx.union(*[tx.union(*[list(sp.sympify(fform_dict[k][n2i(m, M)]).free_symbols) for m in np.arange(-M, M+1)]) for k in fform_dict.keys()])
    return free_symbols


def sanitize_free_symbols(free_symbols):
    ans = free_symbols[:]
    ans.sort(key=lambda x: str(x))
    return ans


def fform_dict_subs(fform_dict, subs_array, M=16):
    subs_fform_dict = {}
    for k in fform_dict.keys():
        subs_fform_dict[k] = np.zeros(shape=(2*M+1,), dtype=object)
        for m in np.arange(-M, M+1):
            subs_fform_dict[k][n2i(m)] = sp.sympify(fform_dict[k][n2i(m)]).subs(subs_array)
    return subs_fform_dict
