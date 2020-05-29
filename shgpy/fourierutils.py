import numpy as np
import sympy as sp
from .shg_symbols import *
from . import core as util
from scipy.integrate import quad
from . import tensorutils as tx


def n2i(n, M=16):
    return n+M


def convolve_tensor_lists(nR1, nR2, M=16, dtype=object):
    test_prod = tx.tensor_product(nR1[0], nR2[0])
    ans = np.zeros(dtype=dtype, shape=(2*M+1,)+test_prod.shape)
    for n in np.arange(-M, M+1):
        for m in np.arange(-M, M+1):
            try:
                ans[n2i(n, M)] += tx.tensor_product(nR1[n2i(m, M)], nR2[n2i(n-m, M)])
            except IndexError:
                pass
    return ans


def formula_from_fexpr(t, M=16):
    expr = 0
    for m in np.arange(-M, M+1):
        expr += t[n2i(m)]*(sp.cos(m*phi)+1j*sp.sin(m*phi))
    return expr


def data_dft(data_dict, M=16):
    ans = {pc:np.zeros(2*M+1, dtype=object) for pc in data_dict.keys()}
    for k in ans.keys():
        for m in np.arange(-M, M+1):
            ydata = np.array(data_dict[k][1])
            xdata = np.linspace(0, 2*np.pi, len(ydata), endpoint=False)
            dx = xdata[1]-xdata[0]
            ans[k][n2i(m, M)] = sum([1/2/np.pi*dx*ydata[i]*np.exp(-1j*m*xdata[i]) for i in range(len(xdata))])
    return ans


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


def load_data_and_fourier_transform(prefix, numbers, suffix, M=16, min_subtract=False, scale=1, d_prefix=None, d_numbers=None, d_suffix=None):
    filenames = [prefix+number+suffix for number in numbers]
    data_dict = plotutils.load_data(filenames)
    if d_prefix is not None and d_numbers is not None and d_suffix is not None:
        d_filenames = [d_prefix+d_number+d_suffix for d_number in d_numbers]
        d_datadict = plotutils.load_data(d_filenames)
        data_dict = util.dark_subtract_dicts(data_dict, d_datadict)
    for k in data_dict.keys():
        for i in range(len(data_dict[k][1])):
            data_dict[k][1][i] = data_dict[k][1][i]*scale
    xdata = np.linspace(0, 2*np.pi, len(data_dict['PP'][0]), endpoint=False)
    for k in data_dict.keys():
        data_dict[k][0] = xdata
        min_data_k = min(data_dict[k][1])
        if min_subtract is True:
            data_dict[k][1] = [data_dict[k][1][i] - min_data_k for i in range(len(data_dict[k][1]))]
    fdata_dict = data_dft(data_dict, M=M)

    for k in data_dict.keys():
        data_dict[k][0] = xdata

    return data_dict, fdata_dict


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
