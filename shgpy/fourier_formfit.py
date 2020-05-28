import sympy as sp
import numpy as np
import shgpy.utilities as util
from shgpy.shg_symbols import *
import shgpy.tensorutils as tx
import shgpy.fourierutils as fx
from scipy.optimize import leastsq
from scipy.optimize import basinhopping
import time
import itertools
from warnings import warn
from scipy.optimize import OptimizeResult
from sympy.utilities.lambdify import lambdastr
import math

def I_component(expr):
    return (expr-expr.subs(sp.I, 0)).subs(sp.I, 1)

def no_I_component(expr):
    return expr.subs(sp.I, 0)

def leastsq_fit(fform_dict, fdata_dict, guess_dict, M=16):
    
    free_symbols = fx.sanitize_free_symbols(fx.extract_free_symbols_from_fform_dict(fform_dict, M=M))
    
    expr_residual_list = []
    for k in sorted(fform_dict.keys()):
        for m in np.arange(-M, M+1):
            expr_residual_list.append(fform_dict[k][fx.n2i(m, M)]-fdata_dict[k][fx.n2i(m, M)])

    pre_residual = sp.lambdify(free_symbols, expr_residual_list)
    residual = lambda x:np.array(pre_residual(*x)).view(np.double)

    guess = [guess_dict[str(k)] for k in free_symbols]

    ans, covr = leastsq(residual, guess)
    return ans, np.dot(residual(ans), residual(ans))

def basinhopping_fit(fform_dict, fdata_dict, guess_dict, niter=200, M=16, method='BFGS', verbose=True, stepsize=0.5):
    
    free_symbols = fx.sanitize_free_symbols(fx.extract_free_symbols_from_fform_dict(fform_dict, M=M))

    util.oprint(verbose, 'Starting energy function generation.')
    start = time.perf_counter()
    energy_expr_list = []
    for k in sorted(fform_dict.keys()):
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            expr2 = I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    energy_expr = sum(energy_expr_list)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    f_energy = lambda x:pre_f_energy(*x)
    end = time.perf_counter()
    util.oprint(verbose, 'Done with energy function generation. It took %s seconds.' % (end-start))

    
    x0 = [guess_dict[str(k)] for k in free_symbols]
    minimizer_kwargs = {'method':method}
    start = time.perf_counter()
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    end = time.perf_counter()

    return ret,end-start

def basinhopping_fit_with_bounds(fform_dict, fdata_dict, guess_dict, bounds_dict, niter=200, M=16, method='L-BFGS-B', verbose=True, stepsize=0.5):
    
    free_symbols = fx.sanitize_free_symbols(fx.extract_free_symbols_from_fform_dict(fform_dict, M=M))

    util.oprint(verbose, 'Starting energy function generation.')
    start = time.perf_counter()
    energy_expr_list = []
    for k in sorted(fform_dict.keys()):
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            expr2 = I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    energy_expr = sum(energy_expr_list)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    f_energy = lambda x:pre_f_energy(*x)
    end = time.perf_counter()
    util.oprint(verbose, 'Done with energy function generation. It took %s seconds.' % (end-start))

    
    x0 = [guess_dict[str(k)] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[str(k)] for k in free_symbols]
    if bounds_dict is None:
        bounds = None
    minimizer_kwargs = {'method':method, 'bounds':bounds}
    start = time.perf_counter()
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    end = time.perf_counter()

    return ret,end-start

def basinhopping_fit_jac(fform_dict, fdata_dict, guess_dict, niter=200, M=16, method='BFGS', verbose=True, stepsize=0.5):
    
    free_symbols = fx.sanitize_free_symbols(fx.extract_free_symbols_from_fform_dict(fform_dict, M=M))

    util.oprint(verbose, 'Starting energy function generation.')
    start = time.perf_counter()
    energy_expr_list = []
    for k in sorted(fform_dict.keys()):
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            expr2 = I_component(sp.sympify(fform_dict[k][fx.n2i(m, M)] - fdata_dict[k][fx.n2i(m, M)]))
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    energy_expr = sum(energy_expr_list)
    denergy_expr = util.gradient(energy_expr, free_symbols)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    pre_df_energy = sp.lambdify(free_symbols, denergy_expr)
    fdf_energy = lambda x:(pre_f_energy(*x),np.array(pre_df_energy(*x)))
    end = time.perf_counter()
    util.oprint(verbose, 'Done with energy function generation. It took %s seconds.' % (end-start))

    x0 = [guess_dict[str(k)] for k in free_symbols]
    minimizer_kwargs = {'method':method, 'jac':True}
    start = time.perf_counter()
    ret = basinhopping(fdf_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    end = time.perf_counter()

    return ret, end-start
