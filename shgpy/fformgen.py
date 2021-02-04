"""Module for generating Fourier formulas.

This module provides the necessary routines for generating Fourier 
formulas; that is, `ndarrays` of `sympy.Expr` where each element
is a formula for a given Fourier component. The key benefit of working
with Fourier formulas rather than `phi`-space formulas is that all of
the fitting is easier -- instead of fitting a dataset to a complicated
trigonometric formula involving terms like ``cos(3*phi)``,
``sin**4(phi)``, etc., we simply fit the Fourier transform of the data
to the Fourier transform of the oritinal fitting formula. This results
not only in a drastic simplification of the fitting routines, but also
it turns out that computing the Fourier transform of a particular
formula is a problem that can be computed mostly ahead of time.

The way that this plays out in `shgpy` is that we compute the Fourier
transform of the uncontracted part of the response function -- that is,
the part of the response function which does not depend on the particular
form of the susceptbility tensor. This is conveniently the only
part which depends on `phi`. The result is a large (Fourier transformed)
tensor which we can save to a file on your machine using `pickle`. This
is done in :func:`~shgpy.fformgen.generate_uncontracted_fourier_transforms`
and :func:`~shgpy.fformgen.generate_uncontracted_fourier_transforms_symb`,
depending on whether the angle of incidence is hardcoded or a
`sympy.Symbol` instance.

Then, at runtime one can choose the susceptibility tensor they want to
fit (i.e. using :mod:`~shgpy.tensor_definitions`) and simply contract
it (using :func:`~shgpy.fformgen.generate_contracted_fourier_transforms`)
with the pickled formula to produce a Fourier formula. This can then
be manipulated and fitted at will (see :mod:`~shgpy.fformfit`).

See the tutorial for more details.

"""
import sympy as sp
import numpy as np
from scipy.integrate import quad
from .core import (
    rotation_matrix3symb,
    n2i,
    tensor_contract,
    tensor_product,
)
from . import shg_symbols as S
import pickle
import sys
import logging
import time

from .core.utilities import _assert_real_params

_logger = logging.getLogger(__name__)


def _substitute_into_array(expr_array, *subs_tuples):
    ans = np.zeros(shape=expr_array.shape, dtype=object).flatten()
    temp = expr_array.flatten()
    for i in range(len(temp)):
        try:
            ans[i] = temp[i].subs(subs_tuples)
        except AttributeError:
            ans[i] = temp[i]
    return ans.reshape(expr_array.shape)


def _round_complex(z, ndigits):
    return round(np.real(z), ndigits)+1j*round(np.imag(z), ndigits)


def _round_expr(expr, ndigits):
    try:
        expr = expr.expand()
        return expr.xreplace(
            {n:(round(sp.re(n), ndigits)+1j*round(sp.im(n), ndigits))
            for n in expr.atoms(sp.Number)}
        )
    except AttributeError:
        return _round_complex(expr, ndigits)


def _round_complex_tensor(t, ndigits):
    ans = t.flatten()
    for i in range(len(ans)):
        try:
            ans[i] = _round_expr(ans[i], ndigits)
        except AttributeError:
            ans[i] = (round(sp.re(ans[i]), ndigits)
                        + 1j*round(sp.im(ans[i]), ndigits))
    ans = ans.reshape(t.shape)
    return ans


def _conjugate_tensor(tensor):
    ans = np.zeros(len(tensor.flatten()), dtype=object)
    for i,expr in enumerate(tensor.flatten()):
        ans[i] = sp.conjugate(sp.sympify(expr))
    return ans.reshape(tensor.shape)


def _convolve_ftensors(nR1, nR2, M=16, dtype=object):
    test_prod = tensor_product(nR1[0], nR2[0])
    ans = np.zeros(dtype=dtype, shape=(2*M+1,)+test_prod.shape)
    for n in np.arange(-M, M+1):
        for m in np.arange(-M, M+1):
            try:
                ans[n2i(n, M)] += tensor_product(
                    nR1[n2i(m, M)],
                    nR2[n2i(n-m, M)],
                )
            except IndexError:
                pass
    return ans


def _load_pickle(filename):
    return np.load(filename, allow_pickle=True)


def _save_fform_dict(filename, _fform_dict):
    str_fform_dict = {
        k:np.array([sp.srepr(s) for s in v])
        for k,v in _fform_dict.items()
    }
    with open(filename, 'wb') as fh:
        pickle.dump(str_fform_dict, fh)


def _load_fform_dict(filename):
    with open(filename, 'rb') as fh:
        str_fform_dict = pickle.load(fh)
    _fform_dict = {
        k:np.array([sp.sympify(s) for s in v])
        for k,v in str_fform_dict.items()
    }
    return _fform_dict


def generate_uncontracted_fourier_transforms(aoi,
                                             uncontracted_filename_prefix,
                                             M=16):
    """Generate uncontracted Fourier transforms.

    Parameters
    ----------
    aoi : float
        Angle of incidence (in radians). This is hardcoded in this
        function but can be left as a free parameter in
        :func:`~shgpy.fformgen.generate_uncontracted_fourier_transforms_symb`.
    uncontracted_filename_prefix : str
        Filename prefix for where to save the pickled Fourier
        transforms. The pickle objects are saved to
        ``uncontracted_filename_prefix+'_pp'``,
        ``uncontracted_filename_prefix+'_ps'``, etc.
    M : int, optional
        Number of Fourier frequencies to compute. Default is 16.
        The Fourier formulas produced with this function have ``2*M+1``
        elements each.

    Notes
    -----
    Saves uncontracted Fourier transforms using `pickle` to the
    specified filenames.

    """
    include_quadrupole = True
    start = time.time()

    ## 
    ## First define all of the types of 
    ## tensors we will need to form the
    ## tensor product.
    ## 
    F = np.array([S.Fx, S.Fy, S.Fz])
    np.set_printoptions(threshold=sys.maxsize)
    R = np.array(rotation_matrix3symb([0, 0, 1], S.phi, ndigits=5))
    Id = np.identity(3)
    k_out = _substitute_into_array(
        np.array([-sp.sin(S.theta), 0, -sp.cos(S.theta)]),
        (S.theta, aoi),
    )
    k_in = _substitute_into_array(
        np.array([-sp.sin(S.theta), 0, sp.cos(S.theta)]),
        (S.theta, aoi),
    )
    proj = Id - tensor_product(k_out, k_out)
    proj_x = proj[0]
    proj_y = proj[1]
    proj_z = proj[2]

    ##
    ## Now take the fourier transform
    ## of all the tensors above. For
    ## tensors which are independent of
    ## phi, of course the fourier transform
    ## just has an m=0 component.
    ##
    rproj = np.zeros(shape=(2*M+1,)+proj.shape, dtype=object)
    rproj[n2i(0, M)] = proj
    rproj_x = np.zeros(shape=(2*M+1,)+proj_x.shape, dtype=object)
    rproj_x[n2i(0, M)] = proj_x
    rproj_y = np.zeros(shape=(2*M+1,)+proj_y.shape, dtype=object)
    rproj_y[n2i(0, M)] = proj_y
    rproj_z = np.zeros(shape=(2*M+1,)+proj_z.shape, dtype=object)
    rproj_z[n2i(0, M)] = proj_z
    rR = np.zeros(shape=(2*M+1, 3, 3), dtype=object)
    rR[n2i(-1, M)] = np.array(
        [
            [0.5, -0.5j, 0.0],
            [0.5j, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=object,
    )
    rR[n2i(0, M)] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=object,
    )
    rR[n2i(1, M)] = np.array(
        [
            [0.5, 0.5j, 0.0],
            [-0.5j, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=object,
    )
    rF = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rF[n2i(0, M)] = F
    rFc = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rFc[n2i(0, M)] = sp.conjugate(F)
    r1_x = _convolve_ftensors(rproj_x, rR)
    r1_y = _convolve_ftensors(rproj_y, rR)
    r1_z = _convolve_ftensors(rproj_z, rR)
    rk_out = np.zeros(shape=(2*M+1,)+k_out.shape, dtype=object)
    rk_out[n2i(0, M)] = k_out
    rk_in = np.zeros(shape=(2*M+1,)+k_in.shape, dtype=object)
    rk_in[n2i(0, M)] = k_in

    ## 
    ## This is so I can address the P and
    ## S components individually.
    ## 
    r1_0 = np.zeros(shape=r1_x.shape, dtype=object)
    r1_s = np.array(
        [
            np.array([r1_0[n2i(m, M)], r1_y[n2i(m, M)], r1_0[n2i(m, M)]])
            for m in np.arange(-M, M+1)
        ],
    )
    r1_p = np.array(
        [
            np.array([r1_x[n2i(m, M)], r1_0[n2i(m, M)], r1_z[n2i(m, M)]])
            for m in np.arange(-M, M+1)
        ],
    )

    ##
    ## Now I am ready to do the long tensor
    ## contraction. I have found that the
    ## computation is much faster if you
    ## resort to small tensor products and build
    ## up the long contraction piece by piece.
    ## In the future I would like to build
    ## a method to do all of this with one
    ## function call, but for now this will do...
    ##
    h7_arr_term1 = []
    h7_arr_term2 = []
    h7_arr_term3 = []
    h7_arr_term4 = []
    for r1 in [r1_p, r1_s]:
        h1 = np.array(
            [
                tensor_contract(r1[n2i(m, M)], [[1, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h1 done.')
        r2 = _convolve_ftensors(h1, h1)
        h2 = np.array(
            [
                tensor_contract(r2[n2i(m, M)], [[0, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h2 done.')
        r3 = _convolve_ftensors(rR, rF)
        h3 = np.array(
            [
                tensor_contract(r3[n2i(m, M)], [[0, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h3 done.')
        h4 = _convolve_ftensors(h2, h3)
        _logger.debug('h4 done.')
        h5 = _convolve_ftensors(h4, h3)
        _logger.debug('h5 done.')
        h6 = _convolve_ftensors(h5, h3)
        _logger.debug('h6 done.')
        h7 = _convolve_ftensors(h6, h3)
        _logger.debug('h7 done.')
        h7_arr_term1.append(h7)
        if include_quadrupole is True:
            r4 = _convolve_ftensors(rR, -1j*rk_in)
            h8 = np.array(
                [
                    tensor_contract(r4[n2i(m, M)], [[0, 2]])
                    for m in np.arange(-M, M+1)
                ]
            )
            _logger.debug('h8 done.')
            h9 = _convolve_ftensors(h7, h8)
            _logger.debug('h9 done.')
            h7_arr_term2.append(h9)
            h7_arr_term3.append(-1*h9)
            h10 = _convolve_ftensors(h9, -1*h8)
            _logger.debug('h10 done.')
            h7_arr_term4.append(h10)
        
    ##
    ## Now I form four of these tensors, each
    ## pertaining to a particular polarization
    ## combination.
    ##
    list_of_terms = [h7_arr_term1, h7_arr_term2, h7_arr_term3, h7_arr_term4]
    nterms = len(list_of_terms)
    _logger.debug('Started substitution.')

    h7_pp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_pp[i] = _substitute_into_array(
            h7_arr[0],
            (S.Fx, np.cos(aoi)),
            (S.Fy, 0),
            (S.Fz, np.sin(aoi)),
        )
    _logger.debug('done 1.')

    h7_ps = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ps[i] = _substitute_into_array(
            h7_arr[1],
            (S.Fx, np.cos(aoi)),
            (S.Fy, 0),
            (S.Fz, np.sin(aoi)),
        )
    _logger.debug('done 2.')
    
    h7_sp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_sp[i] = _substitute_into_array(
            h7_arr[0],
            (S.Fx, 0),
            (S.Fy, -1),
            (S.Fz, 0),
        )
    _logger.debug('done 3.')

    h7_ss = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ss[i] = _substitute_into_array(
            h7_arr[1],
            (S.Fx, 0),
            (S.Fy, -1),
            (S.Fz, 0),
        )
    _logger.debug('done 4.')

    _logger.debug('Generation of uncontracted fourier transforms completed.'
                  f'It took {time.time()-start} seconds.')
    
    ##
    ## Now to generate the fourier transformed
    ## SHG formulae for a particular tensor chi,
    ## all you need to do is to contract these
    ## tensors in a certain way with chi x chi.
    ##
    ## All that's left is to save these h7 tensors
    ## to a file.
    ##
    np.save(uncontracted_filename_prefix+'_pp', h7_pp)
    np.save(uncontracted_filename_prefix+'_ps', h7_ps)
    np.save(uncontracted_filename_prefix+'_sp', h7_sp)
    np.save(uncontracted_filename_prefix+'_ss', h7_ss)


def generate_uncontracted_fourier_transforms_symb(uncontracted_filename_prefix,
                                                  M=16):
    """Generate uncontracted Fourier transforms with symbolic angle of incidence.

    Parameters
    ----------
    uncontracted_filename_prefix : str
        Filename prefix for where to save the pickled Fourier
        transforms. The pickle objects are saved to
        ``uncontracted_filename_prefix+'_pp'``,
        ``uncontracted_filename_prefix+'_ps'``, etc.
    M : int, optional
        Number of Fourier frequencies to compute. Default is 16.
        The Fourier formulas produced with this function have ``2*M+1``
        elements each.

    Notes
    -----
    Saves uncontracted Fourier transforms using `pickle` to the
    specified filenames. The angle of incidence is hardcoded to
    be the `sympy.Symbol` `theta` defined in :mod:`~shgpy.shg_symbols`.

    """
    start = time.time()
    include_quadrupole = True

    ## 
    ## First define all of the types of 
    ## tensors we will need to form the
    ## tensor product.
    ## 
    F = np.array([S.Fx, S.Fy, S.Fz])
    np.set_printoptions(threshold=sys.maxsize)
    R = np.array(rotation_matrix3symb([0, 0, 1], S.phi, ndigits=5))
    Id = np.identity(3)
    k_out = np.array([-sp.sin(S.theta), 0, -sp.cos(S.theta)])
    k_in = np.array([-sp.sin(S.theta), 0, sp.cos(S.theta)])
    proj = Id - tensor_product(k_out, k_out)
    proj_x = proj[0]
    proj_y = proj[1]
    proj_z = proj[2]

    ##
    ## Now take the fourier transform
    ## of all the tensors above. For
    ## tensors which are independent of
    ## phi, of course the fourier transform
    ## just has an m=0 component.
    ##
    rproj = np.zeros(shape=(2*M+1,)+proj.shape, dtype=object)
    rproj[n2i(0, M)] = proj
    rproj_x = np.zeros(shape=(2*M+1,)+proj_x.shape, dtype=object)
    rproj_x[n2i(0, M)] = proj_x
    rproj_y = np.zeros(shape=(2*M+1,)+proj_y.shape, dtype=object)
    rproj_y[n2i(0, M)] = proj_y
    rproj_z = np.zeros(shape=(2*M+1,)+proj_z.shape, dtype=object)
    rproj_z[n2i(0, M)] = proj_z
    rR = np.zeros(shape=(2*M+1, 3, 3), dtype=object)
    rR[n2i(-1, M)] = np.array(
        [
            [0.5, -0.5j, 0.0],
            [0.5j, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=object,
    )
    rR[n2i(0, M)] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=object,
    )
    rR[n2i(1, M)] = np.array(
        [
            [0.5, 0.5j, 0.0],
            [-0.5j, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=object,
    )
    rF = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rF[n2i(0, M)] = F
    rFc = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rFc[n2i(0, M)] = sp.conjugate(F)
    r1_x = _convolve_ftensors(rproj_x, rR)
    r1_y = _convolve_ftensors(rproj_y, rR)
    r1_z = _convolve_ftensors(rproj_z, rR)
    rk_out = np.zeros(shape=(2*M+1,)+k_out.shape, dtype=object)
    rk_out[n2i(0, M)] = k_out
    rk_in = np.zeros(shape=(2*M+1,)+k_in.shape, dtype=object)
    rk_in[n2i(0, M)] = k_in

    ## 
    ## This is so I can address the P and
    ## S components individually.
    ## 
    r1_0 = np.zeros(shape=r1_x.shape, dtype=object)
    r1_s = np.array(
        [
            np.array([r1_0[n2i(m, M)], r1_y[n2i(m, M)], r1_0[n2i(m, M)]])
            for m in np.arange(-M, M+1)
        ],
    )
    r1_p = np.array(
        [
            np.array([r1_x[n2i(m, M)], r1_0[n2i(m, M)], r1_z[n2i(m, M)]])
            for m in np.arange(-M, M+1)
        ],
    )

    ##
    ## Now I am ready to do the long tensor
    ## contraction. I have found that the
    ## computation is much faster if you
    ## resort to small tensor products and build
    ## up the long contraction piece by piece.
    ## In the future I would like to build
    ## a method to do all of this with one
    ## function call, but for now this will do...
    ##
    h7_arr_term1 = []
    h7_arr_term2 = []
    h7_arr_term3 = []
    h7_arr_term4 = []
    for r1 in [r1_p, r1_s]:
        h1 = np.array(
            [
                tensor_contract(r1[n2i(m, M)], [[1, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h1 done.')
        r2 = _convolve_ftensors(h1, h1)
        h2 = np.array(
            [
                tensor_contract(r2[n2i(m, M)], [[0, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h2 done.')
        r3 = _convolve_ftensors(rR, rF)
        h3 = np.array(
            [
                tensor_contract(r3[n2i(m, M)], [[0, 2]])
                for m in np.arange(-M, M+1)
            ],
        )
        _logger.debug('h3 done.')
        h4 = _convolve_ftensors(h2, h3)
        _logger.debug('h4 done.')
        h5 = _convolve_ftensors(h4, h3)
        _logger.debug('h5 done.')
        h6 = _convolve_ftensors(h5, h3)
        _logger.debug('h6 done.')
        h7 = _convolve_ftensors(h6, h3)
        _logger.debug('h7 done.')
        h7_arr_term1.append(h7)
        if include_quadrupole is True:
            r4 = _convolve_ftensors(rR, -1j*rk_in)
            h8 = np.array(
                [
                    tensor_contract(r4[n2i(m, M)], [[0, 2]])
                    for m in np.arange(-M, M+1)
                ],
            )
            _logger.debug('h8 done.')
            h9 = _convolve_ftensors(h7, h8)
            _logger.debug('h9 done.')
            h7_arr_term2.append(h9)
            h7_arr_term3.append(-1*h9)
            h10 = _convolve_ftensors(h9, -1*h8)
            _logger.debug('h10 done.')
            h7_arr_term4.append(h10)
        
    ##
    ## Now I form four of these tensors, each
    ## pertaining to a particular polarization
    ## combination.
    ##
    list_of_terms = [h7_arr_term1, h7_arr_term2, h7_arr_term3, h7_arr_term4]
    nterms = len(list_of_terms)
    _logger.debug('Started substitution.')

    h7_pp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_pp[i] = _substitute_into_array(
            h7_arr[0],
            (S.Fx, sp.cos(S.theta)),
            (S.Fy, 0),
            (S.Fz, sp.sin(S.theta)),
        )
    _logger.debug('done 1.')

    h7_ps = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ps[i] = _substitute_into_array(
            h7_arr[1],
            (S.Fx, sp.cos(S.theta)),
            (S.Fy, 0),
            (S.Fz, sp.sin(S.theta)),
        )
    _logger.debug('done 2.')
    
    h7_sp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_sp[i] = _substitute_into_array(
            h7_arr[0],
            (S.Fx, 0),
            (S.Fy, -1),
            (S.Fz, 0),
        )
    _logger.debug('done 3.')

    h7_ss = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ss[i] = _substitute_into_array(
            h7_arr[1],
            (S.Fx, 0),
            (S.Fy, -1),
            (S.Fz, 0),
        )
    _logger.debug('done 4.')

    _logger.debug('Generation of uncontracted fourier transforms completed. '
                  f'It took {time.time()-start} seconds.')
    
    ##
    ## Now to generate the fourier transformed
    ## SHG formulae for a particular tensor chi,
    ## all you need to do is to contract these
    ## tensors in a certain way with chi x chi.
    ##
    ## All that's left is to save these h7 tensors
    ## to a file.
    ##
    np.save(uncontracted_filename_prefix+'_pp', h7_pp)
    np.save(uncontracted_filename_prefix+'_ps', h7_ps)
    np.save(uncontracted_filename_prefix+'_sp', h7_sp)
    np.save(uncontracted_filename_prefix+'_ss', h7_ss)


def generate_contracted_fourier_transforms_complex(*args, **kwargs):
    raise NotImplementedError('generate_contracted_fourier_transforms_complex'
                              ' was deprecated in version 0.7.0. Use '
                              'generate_contracted_fourier_transforms.')


def generate_contracted_fourier_transforms(save_filename,
                                           uncontracted_filename_prefix,
                                           chi_dipole, chi_quadrupole,
                                           M=16, ndigits=None):
    """Generate contracted Fourier transforms.

    Parameters
    ----------
    save_filename : str or file object
        Filename to save the pickled Fourier formula to.
    uncontracted_filename_prefix : str
        The filename prefix of the pickled uncontracted Fourier
        transforms generated by
        :func:`~shgpy.fformgen.generate_uncontracted_fourier_transforms`.    
    chi_dipole : ndarray
        Dipole susceptibility tensor
    chi_quadupole : ndarray
        Quadrupole susceptibility tensor
    M : int, optional
        Number of Fourier frequencies to compute. Default is 16.
        The Fourier formulas produced with this function have ``2*M+1``
        elements each.
    ndigits : int, optional
        Number of digits to round to for each term in the Fourier formula.
        If `None`, no rounding is done. Default is `None`.

    Raises
    ------
    ValueError
        If not all parameters of `chi_dipole` and `chi_quadrupole`
        hardcoded as real, i.e. you must specifically indicate the
        real and imaginary parts of each variable in `chi_dipole` and
        `chi_quadrupole` (see :func:`~shgpy.core.utilities.make_tensor_complex`
        and :func:`~shgpy.core.utilities.make_tensor_real`).

    Notes
    -----
    Saves contracted Fourier transforms using `pickle` to `save_filename`. 

    """
    ##
    ## First we check if all the parameters in 
    ## chi_dipole and chi_quadrupole are real.
    ## If not, raise an error.
    ##
    for chi in [chi_dipole, chi_quadrupole]:
        _assert_real_params(chi)

    ##
    ## Now I build a useful set of arrays
    ## which we will use later to reduce
    ## the number of lines in this program
    ##
    pcs = ['PP', 'PS', 'SP', 'SS']
    terms_dict = {
        pc:_load_pickle(uncontracted_filename_prefix+'_'+pc.lower()+'.npy')
        for pc in pcs
    }

    contraction_lists_1 = [[[0, 6], [2, 7], [3, 8]],
                           [[0, 7], [2, 8], [3, 9]],
                           [[0, 7], [2, 9], [3, 10], [6, 8]],
                           [[0, 8], [2, 10], [3, 11], [6, 9]]]
    contraction_lists_2 = [[[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]],
                           [[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]]]

    chi_list_1 = [chi_dipole, chi_dipole, chi_quadrupole, chi_quadrupole]
    chi_list_2 = [
        _conjugate_tensor(chi_dipole),
        _conjugate_tensor(chi_quadrupole),
        _conjugate_tensor(chi_dipole),
        _conjugate_tensor(chi_quadrupole),
    ]

    _logger.debug('Finished preparation.')

    ##
    ## Now we do the contraction with chi x chi,
    ## for each polarization combination and for
    ## each term. Each of the tensors h7_pp, ...
    ## has 4 elements, correponsing to the four
    ## terms in |P+ikQ|^2.
    ##
    _fform_dict = {}
    for pc,h7_pc in terms_dict.items():
        _fform_dict[pc] = np.zeros(shape=(2*M+1,), dtype=object)
        for term in range(len(h7_pc)):
            t8_pc_term = np.array(
                [
                    tensor_contract(
                        tensor_product(
                            h7_pc[term][n2i(m, M)],
                            chi_list_1[term],
                        ),
                        contraction_lists_1[term],
                    )
                    for m in np.arange(-M, M+1)
                ],
            )
            t9_pc_term = np.array(
                [
                    tensor_contract(
                        tensor_product(
                            t8_pc_term[n2i(m, M)],
                            chi_list_2[term],
                        ),
                        contraction_lists_2[term],
                    ) 
                    for m in np.arange(-M, M+1)
                ],
            )
            _fform_dict[pc] += np.copy(t9_pc_term)
            _logger.debug('Finished term %s.' % term)
        if ndigits is not None:
            _fform_dict[pc] = _round_complex_tensor(_fform_dict[pc], ndigits)
        _logger.debug('Finished %s.' % pc)

    _save_fform_dict(save_filename, _fform_dict)
