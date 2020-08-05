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
from sympy.utilities.codegen import CCodeGen
import numpy as np
from .core import n2i
from scipy.optimize import (
    basinhopping,
    least_squares,
    dual_annealing,
    OptimizeResult,
)
import tempfile
import time
import logging
from warnings import warn
import os
import shutil
from pathlib import Path
import ctypes

_logger = logging.getLogger(__name__)


def _check_fform(fform):
    if not fform.get_free_symbols():
        message = (
            'fFormContainer object has no free symbols to use as fitting'
            'parameters. Is your fitting formula actually zero?'
        )
        warn(message)
        ret = OptimizeResult(
            x=np.array([]),
            xdict={},
            success=False,
            status=0,
            message=message,
        )
        return ret


def _make_energy_expr(fform, fdat, free_symbols=None):

    if free_symbols is None:
        free_symbols = fform.get_free_symbols()

    M = fform.get_M()
    energy_expr = 0
    xs = sp.MatrixSymbol('xs', len(free_symbols), 1)
    mapping = {fs:xs[i] for i, fs in enumerate(free_symbols)}
    mapping[sp.I] = 1j
    start = time.time()
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            _logger.debug(f'Computing cost function term pc={k} m={m}')
            expr0 = fform.get_pc(k)[n2i(m, M)] - fdat.get_pc(k)[n2i(m, M)]
            energy_expr += (expr0*sp.conjugate(expr0)).xreplace(mapping)
    _logger.debug('Cost expression evaluation took'
                   f' {time.time()-start} seconds.')

    return energy_expr


def _make_energy_expr_list(fform, fdat, free_symbols=None):

    if free_symbols is None:
        free_symbols = fform.get_free_symbols()

    M = fform.get_M()
    energy_expr_list = []
    xs = sp.MatrixSymbol('xs', len(free_symbols), 1)
    mapping = {fs:xs[i] for i, fs in enumerate(free_symbols)}
    mapping[sp.I] = 1j
    start = time.time()
    for k in fform.get_keys():
        for m in np.arange(-M, M+1):
            _logger.debug(f'Computing cost function term pc={k} m={m}')
            expr0 = fform.get_pc(k)[n2i(m, M)] - fdat.get_pc(k)[n2i(m, M)]
            energy_expr_list.append(
                (expr0*sp.conjugate(expr0)).xreplace(mapping)
            )
    _logger.debug('Cost expression evaluation took'
                   f' {time.time()-start} seconds.')

    return energy_expr_list


def _make_denergy_expr(energy_expr):
    xs = list(energy_expr.free_symbols)[0]
    n = xs.shape[0]
    return np.array([sp.diff(energy_expr, xs[i]) for i in range(n)])


def _fixed_autowrap(energy_expr, prefix, save_filename=None):

    codegen = CCodeGen()

    routines = []
    _logger.debug('Writing code for energy_expr')
    routines.append(
        codegen.routine('autofunc', energy_expr)
    )
    [(c_name, bad_c_code), (h_name, h_code)] = codegen.write(
        routines,
        prefix,
    )
    c_code = '#include <complex.h>\n'
    c_code += bad_c_code
    c_code = c_code.replace('conjugate', 'conj')

    write_directory = Path(tempfile.mkdtemp()).absolute()

    c_path = write_directory / Path(prefix + '.c')
    h_path = write_directory / Path(prefix + '.h')
    o_path = write_directory / Path(prefix + '.o')
    so_path = write_directory / Path(prefix + '.so')

    with open(c_path, 'w') as fh:
        fh.write(c_code)
        fh.write('\n')
    with open(h_path, 'w') as fh:
        fh.write(h_code)
        fh.write('\n')

    start = time.time()
    _logger.debug('Start compiling code.')
    os.system(f'gcc -c -lm {c_path} -o {o_path}')
    os.system(f'gcc -shared {o_path} -o {so_path}')
    _logger.debug(f'Compiling code took {time.time()-start} seconds.')

    if save_filename is not None:
        if Path(save_filename).exists():
            Path(save_filename).unlink()
        shutil.copy(so_path, save_filename)

    cost_func = _load_func(so_path)

    shutil.rmtree(write_directory)

    return cost_func


def _make_energy_func_chunked(energy_expr_list, prefix, save_filename=None):

    codegen = CCodeGen()

    routines = []
    for i, expr in enumerate(energy_expr_list):
        _logger.debug(f'Writing code for expr {i} of {len(energy_expr_list)}')
        routines.append(
            codegen.routine('expr'+str(i), expr)
        )
    [(c_name, bad_c_code), (h_name, h_code)] = codegen.write(
        routines,
        prefix,
    )
    c_code = '#include <complex.h>\n'
    c_code += bad_c_code
    c_code += """
double autofunc(double *xs){

    double autofunc_result;
    autofunc_result = %s;
    return autofunc_result;

}
""" % ('+'.join(['expr'+str(i)+'(xs)' for i in range(len(energy_expr_list))]))
    c_code = c_code.replace('conjugate', 'conj')

    h_code = h_code.replace('\n#endif', """double autofunc(double *xs);

#endif
""")

    write_directory = Path(tempfile.mkdtemp()).absolute()

    c_path = write_directory / Path(prefix + '.c')
    h_path = write_directory / Path(prefix + '.h')
    o_path = write_directory / Path(prefix + '.o')
    so_path = write_directory / Path(prefix + '.so')

    with open(c_path, 'w') as fh:
        fh.write(c_code)
        fh.write('\n')
    with open(h_path, 'w') as fh:
        fh.write(h_code)
        fh.write('\n')

    start = time.time()
    _logger.debug('Start compiling code.')
    os.system(f'gcc -c -lm -fPIC {c_path} -o {o_path}')
    os.system(f'gcc -shared -fPIC {o_path} -o {so_path}')
    _logger.debug(f'Compiling code took {time.time()-start} seconds.')

    if save_filename is not None:
        if Path(save_filename).exists():
            Path(save_filename).unlink()
        shutil.copy(so_path, save_filename)

    cost_func = _load_func(so_path)

    shutil.rmtree(write_directory)

    return cost_func


def _make_energy_func_auto(energy_expr, save_filename=None):
    energy_func = _fixed_autowrap(
        energy_expr,
        'SHGPY_COST_FUNC',
        save_filename,
    )

    return energy_func


def _load_func(load_cost_func_filename):
    start = time.time()
    c_lib = ctypes.CDLL(load_cost_func_filename)
    c_lib.autofunc.restype = ctypes.c_double
    _logger.debug(f'Importing shared library took'
                   f' {time.time()-start} seconds.')

    def cost_func(x):
        c_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return c_lib.autofunc(c_x)

    return cost_func


def _make_energy_func_wrapper(fform, fdat, free_symbols=None,
                              chunk=False, save_filename=None):

    if chunk:
        energy_expr_list = _make_energy_expr_list(
            fform,
            fdat,
            free_symbols,
        )
        energy_func = _make_energy_func_chunked(
            energy_expr_list,
            'SHGPY_COST_FUNC',
            save_filename,
        )
        return energy_func

    else:
        energy_expr = _make_energy_expr(
            fform,
            fdat,
            free_symbols,
        )
        energy_func = _make_energy_func_auto(
            energy_expr,
            save_filename,
        ) 
        return energy_func


def gen_cost_func(fform, fdat, argument_list=None,
                  chunk=False, save_filename=None):
    """Generate a cost function as an .so file and save it to disk.

    Parameters
    ----------
    fform : fFormContainer
        The Fourier formula to use
    fdat : fDataContainer
        The fourier data to fit
    argument_list : array_like of sympy.Symbol, optional
        Specify an order for the arguments. Default is alphabetical by
        str(sympy.Symbol).
    chunk : bool, optional
        Chunk the function generation into multiple steps (one for each
        Fourier component in fform and fdat). Default is False.
    save_filename : path_like, optional
        Filename to save the result. Default is not to save.

    Returns
    -------
    cost_func : function
        Result after compiling generated C code.

    """
    return _make_energy_func_wrapper(fform, fdat, free_symbols=argument_list,
                                     chunk=chunk, save_filename=save_filename)
        
        
def _make_denergy_func_auto(denergy_expr, save_filename_prefix=None):
    funcs = []
    for i, expr in enumerate(denergy_expr):
        if save_filename_prefix is None:
            new_func = _fixed_autowrap(expr, 'SHGPY_COST_FUNC')
        else:
            new_func = _fixed_autowrap(expr, 'SHGPY_COST_FUNC',
                            Path(save_filename_prefix+str(i)+'.so'))
        funcs.append(new_func)
    return lambda x: np.array([func(x) for func in funcs])


def _make_energy_and_denergy_func_wrapper(fform, fdat, free_symbols=None,
                                           chunk=False, save_filename=None,
                                           grad_save_filename_prefix=None):

    if chunk:
        raise NotImplementedError('Chunking the gradient function is not'
                                  ' implemented yet.')
    else:
        energy_expr = _make_energy_expr(fform, fdat, free_symbols)
        denergy_expr = _make_denergy_expr(energy_expr)
        f_energy = _make_energy_func_auto(
            energy_expr,
            save_filename,
        )
        df_energy = _make_denergy_func_auto(
            denergy_expr,
            grad_save_filename_prefix,
        )
        fdf_energy = lambda x: (f_energy(x), df_energy(x))

    return fdf_energy


def _load_energy_and_denergy_func(load_cost_func_filename,
                                  load_grad_cost_func_filename_prefix):
    f_energy = _load_func(load_cost_func_filename)
    funcs = []
    i = 0
    while True:
        new_filename = load_grad_cost_func_filename_prefix+str(i)+'.so'
        if Path(new_filename).exists():
            funcs.append(
                _load_func(load_grad_cost_func_filename_prefix+str(i)+'.so')
            )
            i += 1
        else:
            _logger.debug('Loaded {i} gradient functions.')
            break
    df_energy = lambda x: np.array([func(x) for func in funcs])
    fdf_energy = lambda x: (f_energy(x), df_energy(x))
    return fdf_energy


def least_squares_fit(fform, fdat, guess_dict,
                      chunk_cost_func=False, save_cost_func_filename=None,
                      load_cost_func_filename=None, least_sq_kwargs={}):
    """No nonsense least-squares fit of RA-SHG data.

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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.
    least_sq_kwargs : dict, optional
        Dictionary of additional options to pass to
        scipy.optimize.least_squares. Default is ``{}``.

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

    if load_cost_func_filename is not None:
        pre_f_energy = _load_func(load_cost_func_filename)
        f_energy = lambda x: np.sqrt(pre_f_energy(x))
    else:
        pre_f_energy = _make_energy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
        )
        f_energy = lambda x: np.sqrt(pre_f_energy(x))

    _logger.info('Done with energy function generation. It took'
                 f'{time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]

    _logger.info('Starting least squares minimization.')
    start = time.time()
    ret = least_squares(f_energy, x0, **least_sq_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with least squares minimization. It took'
                 f'{ret.time} seconds.')

    return ret


def least_squares_fit_with_bounds(fform, fdat, guess_dict,
                                  bounds_dict, chunk_cost_func=False,
                                  save_cost_func_filename=None,
                                  load_cost_func_filename=None,
                                  least_sq_kwargs={}):
    """No nonsense least-squares fit of RA-SHG data.

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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.
    least_sq_kwargs : dict, optional
        Dictionary of additional options to pass to
        scipy.optimize.least_squares. Default is ``{}``.

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

    if load_cost_func_filename is not None:
        pre_f_energy = _load_func(load_cost_func_filename)
        f_energy = lambda x: np.sqrt(pre_f_energy(x))
    else:
        pre_f_energy = _make_energy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
        )
        f_energy = lambda x: np.sqrt(pre_f_energy(x))

    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    bounds = [
        [bounds_dict[k][0] for k in free_symbols],
        [bounds_dict[k][1] for k in free_symbols],
    ]

    _logger.info('Starting least squares minimization.')
    start = time.time()
    ret = least_squares(f_energy, x0, bounds=bounds, **least_sq_kwargs)
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with least squares minimization. It took '
                 f'{ret.time} seconds.')

    return ret


def basinhopping_fit(fform, fdat, guess_dict, niter, method='BFGS',
                     args=(), stepsize=0.5, basinhopping_kwargs={},
                     chunk_cost_func=False, save_cost_func_filename=None,
                     load_cost_func_filename=None):
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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.

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

    if load_cost_func_filename is not None:
        f_energy = _load_func(load_cost_func_filename)
    else:
        f_energy = _make_energy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
        )

    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]

    minimizer_kwargs = {'method':method, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(
        f_energy,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        stepsize=stepsize,
        **basinhopping_kwargs,
    )
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with basinhopping minimization. It took '
                 f'{ret.time} seconds.')

    return ret


def basinhopping_fit_with_bounds(fform, fdat, guess_dict, bounds_dict,
                                 niter, method='L-BFGS-B', args=(),
                                 stepsize=0.5, basinhopping_kwargs={},
                                 chunk_cost_func=False,
                                 save_cost_func_filename=None,
                                 load_cost_func_filename=None):
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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.

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
 
    if load_cost_func_filename is not None:
        f_energy = _load_func(load_cost_func_filename)
    else:
        f_energy = _make_energy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
        )

    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

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
    ret = basinhopping(
        f_energy,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        stepsize=stepsize,
        **basinhopping_kwargs,
    )
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with basinhopping minimization. It took '
                 f'{ret.time} seconds.')

    return ret


def basinhopping_fit_jac(fform, fdat, guess_dict, niter, method='BFGS',
                         args=(), stepsize=0.5, basinhopping_kwargs={},
                         chunk_cost_func=False, save_cost_func_filename=None,
                         grad_save_cost_func_filename_prefix=None,
                         load_cost_func_filename=None,
                         load_grad_cost_func_filename_prefix=None):
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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    grad_save_cos_func_filename : path-like, optional
        If provided, save the gradient cost functions as shared libraries
        at the locations defined by ``...0.so``, ``...1.so``, etc.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.
        Must be used in tandem with ``load_grad_cost_func_filename_prefix``.
    load_grad_cost_func_filename_prefix : path-like, optional
        If provided, load the gradient functions defined by this prefix.
        Must be used in tandem with ``load_cost_func_filename``.

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

    if (load_cost_func_filename is not None
            and load_grad_cost_func_filename_prefix is not None):
        fdf_energy = _load_energy_and_denergy_func(
            load_cost_func_filename,
            load_grad_cost_func_filename_prefix,
        )
            
    else:
        fdf_energy = _make_energy_and_denergy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
            grad_save_cost_func_filename_prefix,
        )
    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]

    minimizer_kwargs = {'method':method, 'jac':True, 'args':args}
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(
        fdf_energy,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        stepsize=stepsize,
        **basinhopping_kwargs,
    )
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with basinhopping minimization. It took '
                 f'{ret.time} seconds.')

    return ret


def basinhopping_fit_jac_with_bounds(fform, fdat, guess_dict, bounds_dict,
                                     niter, method='L-BFGS-B', args=(),
                                     stepsize=0.5, basinhopping_kwargs={},
                                     chunk_cost_func=False, 
                                     save_cost_func_filename=None,
                                     grad_save_cost_func_filename_prefix=None,
                                     load_cost_func_filename=None,
                                     load_grad_cost_func_filename_prefix=None):
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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    grad_save_cost_func_filename_prefix : path-like, optional
        If provided, save the gradient cost functions as shared libraries
        at the locations defined by ``...0.so``, ``...1.so``, etc.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.
        Must be used in tandem with ``load_grad_cost_func_filename_prefix``.
    load_grad_cost_func_filename_prefix : path-like, optional
        If provided, load the gradient functions defined by this prefix.
        Must be used in tandem with ``load_cost_func_filename``.

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

    if (load_cost_func_filename is not None
            and load_grad_cost_func_filename_prefix is not None):
        fdf_energy = _load_energy_and_denergy_func(
            load_cost_func_filename,
            load_grad_cost_func_filename_prefix,
        )

    else:
        fdf_energy = _make_energy_and_denergy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
            grad_save_cost_func_filename_prefix,
        )
    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

    x0 = [guess_dict[k] for k in free_symbols]
    if bounds_dict is not None:
        bounds = [bounds_dict[k] for k in free_symbols]
    if bounds_dict is None:
        bounds = None

    minimizer_kwargs = {
        'method':method,
        'jac':True,
        'bounds':bounds,
        'args':args,
    }
    if 'minimizer_kwargs' in basinhopping_kwargs.keys():
        minimizer_kwargs.update(
            basinhopping_kwargs.pop('minimizer_kwargs')
        )

    _logger.info('Starting basinhopping minimization.')
    start = time.time()
    ret = basinhopping(
        fdf_energy,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        stepsize=stepsize,
        **basinhopping_kwargs,
    )
    ret.time = time.time()-start
    ret.xdict = {k:ret.x[i] for i,k in enumerate(free_symbols)}
    _logger.info('Done with basinhopping minimization. It took '
                 f'{ret.time} seconds.')

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
    callback=None,
    chunk_cost_func=False,
    save_cost_func_filename=None,
    load_cost_func_filename=None,
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
    chunk_cost_func : bool, optional
        Whether to chunk the cost function generation into multiple parts.
        Default is False.
    save_cost_func_filename : path-like, optional
        If provided, save the cost function (as a shared library) at this
        location.
    load_cost_func_filename : path-like, optional
        If provided, load the cost function at this location.

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

    if load_cost_func_filename is not None:
        f_energy = _load_func(load_cost_func_filename)

    else:
        f_energy = _make_energy_func_wrapper(
            fform,
            fdat,
            free_symbols,
            chunk_cost_func,
            save_cost_func_filename,
        )

    _logger.info('Done with energy function generation. It took '
                 f'{time.time()-start} seconds.')

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
