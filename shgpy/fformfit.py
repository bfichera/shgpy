"""Fourier fitting module for fitting RA-SHG data in Fourier space.

This module provides two different methods (up to minor variations) for
fitting RA-SHG data in Fourier space. The first is a simple
least-square method (using `scipy.optimize.least_squares`) which is
extremely fast and works well for simple problems. The second is
a so-called basinhopping method (using `scipy.optimize.basinhopping`)
which is proficient at finding the global minimum of complicated cost
functions with many fitting parameters and local minima. See the
relevant `scipy` documentation for more info.

"""
import sympy as sp
from sympy.utilities.autowrap import autowrap
import numpy as np
from .core import n2i
from scipy.optimize import (
    basinhopping,
    least_squares,
    dual_annealing,
    OptimizeResult,
)
import time
import logging
from warnings import warn

_logger = logging.getLogger(__name__)


def _check_fform(fform):
    if not fform.get_free_symbols():
        message = 'fFormContainer object has no free symbols to use as fitting parameters. Is your fitting formula actually zero?'
        warn(message)
        ret = OptimizeResult(
            x=np.array([]),
            xdict={},
            success=False,
            status=0,
            message=message,
        )
        return ret


def _I_component(expr):
    return (expr-expr.subs(sp.I, 0)).subs(sp.I, 1)


def _no_I_component(expr):
    return expr.subs(sp.I, 0)


def _make_energy_expr(fform, fdat):
    M = fform.get_M()
    energy_expr_list = []
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr1 = _no_I_component(
                fform.get_pc(k)[n2i(m, M)] - fdat.get_pc(k)[n2i(m, M)]
            )
            expr2 = _I_component(
                fform.get_pc(k)[n2i(m, M)] - fdat.get_pc(k)[n2i(m, M)]
            )
            energy_expr_list.append(expr1**2)
            energy_expr_list.append(expr2**2)

    return sum(energy_expr_list)


def _make_denergy_expr(free_symbols, energy_expr):
    def gradient(expr, free_symbols):
        return np.array([sp.diff(expr, fs) for fs in free_symbols])
    return gradient(energy_expr, free_symbols)


def _make_energy_func(free_symbols, energy_expr):
    xs = sp.MatrixSymbol('xs', len(free_symbols), 1)
    mapping = {fs:xs[i] for i, fs in enumerate(free_symbols)}
    easy_energy_expr = energy_expr.xreplace(mapping)
    return autowrap(easy_energy_expr, backend='f2py')


def _make_denergy_func(free_symbols, denergy_expr):
    xs = sp.MatrixSymbol('xs', len(free_symbols), 1)
    mapping = {fs:xs[i] for i, fs in enumerate(free_symbols)}
    easy_denergy_expr = np.array([expr.xreplace(mapping) for expr in denergy_expr])
    funcs = [autowrap(expr, backend='f2py') for expr in easy_denergy_expr]
    return lambda x: np.array([func(x) for func in funcs])


def _make_energy_and_denergy_func(free_symbols, energy_expr, denergy_expr):
    f_energy = _make_energy_func(free_symbols, energy_expr)
    df_energy = _make_denergy_func(free_symbols, denergy_expr)
    fdf_energy = lambda x:(f_energy(x), df_energy(x))
    return fdf_energy


# TODO
# Apply all the autowrap stuff to the residuals here.
def least_squares_fit(fform, fdat, guess_dict):
    """Least-squares fit of RA-SHG data.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    """
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    expr_residual_list = []
    for pc in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr_residual_list.append(fform.get_pc(pc)[n2i(m, M)] - fdat.get_pc(pc)[n2i(m, M)])

    _logger.info('Starting residual function generation.')
    start = time.time()
    pre_residual = sp.lambdify(free_symbols, expr_residual_list)
    residual = lambda x:np.array(pre_residual(*x)).view(np.double)
    _logger.info(f'Done with residual function generation. It took {time.time()-start} seconds.')

    guess = [guess_dict[k] for k in free_symbols]

    _logger.info('Starting least squares minimizations.')
    start = time.time()
    ret = least_squares(residual, guess)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Finished least squares minimization. It took {time.time()-start} seconds.')
    
    return ret


# TODO
# Apply all the autowrap stuff to the residuals here.
def least_squares_fit_with_bounds(fform, fdat, guess_dict, bounds_dict):
    """Least-squares fit of RA-SHG data with bounds.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    bounds_dict : dict
        Dict of form ``{sympy.Symbol:tuple}``. `tuple` should be of form
        ``(lower_bound, upper_bound)``.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()
    M = fform.get_M()

    expr_residual_list = []
    for pc in fform.get_keys():
        for m in np.arange(-M, M+1):
            expr_residual_list.append(fform.get_pc(pc)[n2i(m, M)] - fdat.get_pc(pc)[n2i(m, M)])

    _logger.info('Starting residual function generation.')
    start = time.time()
    pre_residual = sp.lambdify(free_symbols, expr_residual_list)
    residual = lambda x:np.array(pre_residual(*x)).view(np.double)
    _logger.info(f'Done with residual function generation. It took {time.time()-start} seconds.')

    guess = [guess_dict[k] for k in free_symbols]
    bounds = [[bounds_dict[k][0] for k in free_symbols], [bounds_dict[k][1] for k in free_symbols]]

    _logger.info('Starting least squares minimizations.')
    start = time.time()
    ret = least_squares(residual, guess, bounds=bounds)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Finished least squares minimization. It took {time.time()-start} seconds.')
    
    return ret


def basinhopping_fit(fform, fdat, guess_dict, niter, method='BFGS', args=(), stepsize=0.5, basinhopping_kwargs={}):
    """Basinhopping fit of RA-SHG data.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    niter : int
        Number of basinhopping iterations (see scipy documentation)
    method : str, optional
        Minimization method to use, defaults to `'BFGS'`. See scipy
        documentation for more information.
    args : tuple, optional
        Additional arguments to give to the minimizer.
    stepsize : float, optional
        Basinhopping stepsize, defaults to `0.5`. See scipy documentation
        for more information.
    basinhopping_kwargs : dict, optional
        Other options to pass to the basinhopping routine. See scipy
        documentation for more information.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()

    _logger.info('Starting energy function generation.')
    start = time.time()

    energy_expr = _make_energy_expr(fform, fdat)
    f_energy = _make_energy_func(free_symbols, energy_expr)

    _logger.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]

    minimizer_kwargs = {'method':method, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize, **basinhopping_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_with_bounds(fform, fdat, guess_dict, bounds_dict, niter, method='L-BFGS-B', args=(), stepsize=0.5, basinhopping_kwargs={}):
    """Basinhopping fit of RA-SHG data with bounds.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    bounds_dict : dict
        Dict of form ``{sympy.Symbol:tuple}``. `tuple` should be of form
        ``(lower_bound, upper_bound)``.
    niter : int
        Number of basinhopping iterations (see scipy documentation)
    method : str, optional
        Minimization method to use, defaults to `'L-BFGS-B'`. See scipy
        documentation for more information.
    args : tuple, optional
        Additional arguments to give to the minimizer.
    stepsize : float, optional
        Basinhopping stepsize, defaults to `0.5`. See scipy documentation
        for more information.
    basinhopping_kwargs : dict, optional
        Other options to pass to the basinhopping routine. See scipy
        documentation for more information.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()

    _logger.info('Starting energy function generation.')
    start = time.time()
 
    energy_expr = _make_energy_expr(fform, fdat)
    f_energy = _make_energy_func(free_symbols, energy_expr)

    _logger.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None

    minimizer_kwargs = {'method':method, 'bounds':bounds, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    start = time.time()
    _logger.info('Starting basinhopping minimization.')
    ret = basinhopping(f_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize, **basinhopping_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_jac(fform, fdat, guess_dict, niter, method='BFGS', args=(), stepsize=0.5, basinhopping_kwargs={}):
    """Basinhopping fit of RA-SHG data.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    niter : int
        Number of basinhopping iterations (see scipy documentation)
    method : str, optional
        Minimization method to use, defaults to `'BFGS'`. See scipy
        documentation for more information.
    args : tuple, optional
        Additional arguments to give to the minimizer.
    stepsize : float, optional
        Basinhopping stepsize, defaults to `0.5`. See scipy documentation
        for more information.
    basinhopping_kwargs : dict, optional
        Other options to pass to the basinhopping routine. See scipy
        documentation for more information.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    Notes
    -----
    This function computes and supplies the gradient function to the scipy
    basinhopping algorithm. This requires some computational power up front
    but can speed up the minimization algorithm.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()

    _logger.info('Starting energy function generation.')
    start = time.time()

    energy_expr = _make_energy_expr(fform, fdat)
    denergy_expr = _make_denergy_expr(free_symbols, energy_expr)

    fdf_energy = _make_energy_and_denergy_func(free_symbols, energy_expr, denergy_expr)
    _logger.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]

    minimizer_kwargs = {'method':method, 'jac':True, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(fdf_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize, **basinhopping_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def basinhopping_fit_jac_with_bounds(fform, fdat, guess_dict, bounds_dict, niter, method='L-BFGS-B', args=(), stepsize=0.5, basinhopping_kwargs={}):
    """Basinhopping fit of RA-SHG data with bounds.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    bounds_dict : dict
        Dict of form ``{sympy.Symbol:tuple}``. `tuple` should be of form
        ``(lower_bound, upper_bound)``.
    niter : int
        Number of basinhopping iterations (see scipy documentation)
    method : str, optional
        Minimization method to use, defaults to `'L-BFGS-B'`. See scipy
        documentation for more information.
    args : tuple, optional
        Additional arguments to give to the minimizer.
    stepsize : float, optional
        Basinhopping stepsize, defaults to `0.5`. See scipy documentation
        for more information.
    basinhopping_kwargs : dict, optional
        Other options to pass to the basinhopping routine. See scipy
        documentation for more information.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    Notes
    -----
    This function computes and supplies the gradient function to the scipy
    basinhopping algorithm. This requires some computational power up front
    but can speed up the minimization algorithm.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()

    _logger.info('Starting energy function generation.')
    start = time.time()
    energy_expr = _make_energy_expr(fform, fdat)
    denergy_expr = _make_denergy_expr(free_symbols, energy_expr)

    fdf_energy = _make_energy_and_denergy_func(free_symbols, energy_expr, denergy_expr)
    _logger.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None

    minimizer_kwargs = {'method':method, 'jac':True, 'bounds':bounds, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(fdf_energy, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, stepsize=stepsize, **basinhopping_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Done with basinhopping minimization. It took {ret.time} seconds.')

    return ret


def dual_annealing_fit_with_bounds(
    fform,
    fdat,
    guess_dict,
    bounds_dict,
    maxiter=1000,
    local_search_options={},
    initial_temp=5230,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1e7,
    seed=None,
    no_local_search=True,
    callback=None
):
    """Simulated annealing fit of RA-SHG data with bounds.

    Parameters
    ----------
    fform : fFormContainer
        Instance of class :class:`~shgpy.core.data_handler.fFormContainer`.
        This is the (Fourier-transformed) fitting formula.
    fdat : fDataContainer
        Instance of class :class:`~shgpy.core.data_handler.fDataContainer`.
        This is the (Fourier-transformed) data to fit.
    guess_dict : dict
        Dict of form ``{sympy.Symbol:float}``. This is the initial guess.
    bounds_dict : dict
        Dict of form ``{sympy.Symbol:tuple}``. `tuple` should be of form
        ``(lower_bound, upper_bound)``.
    maxiter : int
        Maximum number of global search iterations.
    local_search_options : dict, optional
        Extra keyword arguments to be passed to the local minimizer.
    initial_temp : float, optional
        Initial temperature. Default is 5230.
    restart_temp_ratio : float, optional
        When the temperature reaches ``initial_temp * restart_temp_ratio``,
        the reannealing process is triggered. Default value is 2e-5. Range
        is (0,1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62.
    accept : float, optional
        Parameter for acceptance distribution. Default is -5.0.
    maxfun : int, optional
        Soft limit for the number of objective function calls. Default
        is 1e7.
    seed : {int, RandomState, Generator}, optional
        The random seed to use.
    no_local_search : bool, optional
        If True, perform traditional generalized simulated annealing with
        no local search.
    callback : callable, optional
        A callback function with signature ``callback(x, f, context)``
        which will be called for all minima found.
    x0 : ndarray, shape(n,), optional
        Initial guess.

    Returns
    -------
    ret : scipy.optimize.OptimizeResult
        Instance of class :class:`~scipy.optimize.OptimizeResult`.
        See `scipy` documentation for further description. Includes
        additional attribute ``ret.xdict`` which is a `dict` of 
        ``{sympy.Symbol:float}`` indicating the final answer as
        a dictionary.

    Notes
    -----
    See the ``scipy.optimize.dual_annealing`` documentation for more info.

    """
    check = _check_fform(fform)
    if check:
        return check
    free_symbols = fform.get_free_symbols()

    _logger.info('Starting energy function generation.')
    start = time.time()
    energy_expr = _make_energy_expr(fform, fdat)

    f_energy = _make_energy_func(free_symbols, energy_expr)
    _logger.info(f'Done with energy function generation. It took {time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None

    start = time.time()
    _logger.info('Starting simulated annealing.')
    ret = dual_annealing(
        f_energy,
        bounds,
        maxiter=maxiter,
        local_search_options=local_search_options,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
        seed=seed,
        no_local_search=no_local_search,
        callback=callback,
        x0=x0,
    )
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info(f'Done with simulated annealing. It took {ret.time} seconds.')

    return ret
