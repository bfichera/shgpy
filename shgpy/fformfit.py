import sympy as sp
import numpy as np
from .core import n2i
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
import time
import logging
import logging.config

logging.getLogger(__name__)


def I_component(expr):
    return (expr-expr.subs(sp.I, 0)).subs(sp.I, 1)


def no_I_component(expr):
    return expr.subs(sp.I, 0)


def least_squares_fit(fform, fdat, guess_dict):

    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    expr_residual_list = []
    for pc in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr_residual_list.append(fform.get_pc(pc)[n2i(m, M)] - fdat.get_pc(pc)[n2i(m, M)])

    logging.info('Starting residual function generation.')
    start = time.time()
    pre_residual = sp.lambdify(free_symbols, expr_residual_list)
    residual = lambda x:np.array(pre_residual(*x)).view(np.double)
    logging.info(f'Done with residual function generation. It took {time.time()-start} seconds.')

    guess = [guess_dict[k] for k in free_symbols]

    logging.info('Starting least squares minimizations.')
    start = time.time()
    ret = least_squares(residual, guess)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Finished least squares minimization. It took {time.time()-start} seconds.')
    
    return ret


def least_squares_fit_with_bounds(fform, fdat, guess_dict, bounds_dict):

    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    expr_residual_list = []
    for pc in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr_residual_list.append(fform.get_pc(pc)[n2i(m, M)] - fdat.get_pc(pc)[n2i(m, M)])

    logging.info('Starting residual function generation.')
    start = time.time()
    pre_residual = sp.lambdify(free_symbols, expr_residual_list)
    residual = lambda x:np.array(pre_residual(*x)).view(np.double)
    logging.info(f'Done with residual function generation. It took {time.time()-start} seconds.')

    guess = [guess_dict[k] for k in free_symbols]
    bounds = [[bounds_dict[k][0] for k in free_symbols], [bounds_dict[k][1] for k in free_symbols]]

    logging.info('Starting least squares minimizations.')
    start = time.time()
    ret = least_squares(residual, guess, bounds=bounds)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Finished least squares minimization. It took {time.time()-start} seconds.')
    
    return ret


def basinhopping_fit(fform, fdat, guess_dict, niter, method='BFGS', stepsize=0.5):
    
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    logging.info('Starting energy function generation.')
    start = time.time()
    energy_expr_list = []
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            expr2 = I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    energy_expr = sum(energy_expr_list)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    f_energy = lambda x:pre_f_energy(*x)
    logging.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    minimizer_kwargs = {'method':method}
    logging.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_with_bounds(fform, fdat, guess_dict, bounds_dict, niter, method='L-BFGS-B', stepsize=0.5):
    
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    logging.info('Starting energy function generation.')
    start = time.time()
    energy_expr_list = []
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            expr2 = I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    energy_expr = sum(energy_expr_list)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    f_energy = lambda x:pre_f_energy(*x)
    logging.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None
    minimizer_kwargs = {'method':method, 'bounds':bounds}
    start = time.time()
    logging.info('Starting basinhopping minimization.')
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_jac(fform, fdat, guess_dict, niter, method='BFGS', stepsize=0.5):
    
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    logging.info('Starting energy function generation.')
    start = time.time()
    energy_expr_list = []
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            expr2 = I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    def gradient(expr, free_symbols):
        return np.array([sp.diff(expr, fs) for fs in free_symbols])

    energy_expr = sum(energy_expr_list)
    denergy_expr = gradient(energy_expr, free_symbols)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    pre_df_energy = sp.lambdify(free_symbols, denergy_expr)
    fdf_energy = lambda x:(pre_f_energy(*x),np.array(pre_df_energy(*x)))
    logging.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    minimizer_kwargs = {'method':method, 'jac':True}
    logging.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(fdf_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_jac_with_bounds(fform, fdat, guess_dict, bounds_dict, niter, method='L-BFGS-B', stepsize=0.5):
    
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    logging.info('Starting energy function generation.')
    start = time.time()
    energy_expr_list = []
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr1 = no_I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            expr2 = I_component(fform.get_pc(k)[n2i(m, M)]-fdat.get_pc(k)[n2i(m, M)])
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    def gradient(expr, free_symbols):
        return np.array([sp.diff(expr, fs) for fs in free_symbols])

    energy_expr = sum(energy_expr_list)
    denergy_expr = gradient(energy_expr, free_symbols)

    pre_f_energy = sp.lambdify(free_symbols, energy_expr)
    pre_df_energy = sp.lambdify(free_symbols, denergy_expr)
    fdf_energy = lambda x:(pre_f_energy(*x),np.array(pre_df_energy(*x)))
    logging.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None
    minimizer_kwargs = {'method':method, 'jac':True, 'bounds':bounds}
    logging.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(fdf_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    logging.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret
