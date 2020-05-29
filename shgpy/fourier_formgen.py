import sympy as sp
import numpy as np
from . import core as util
from .shg_symbols import *
from . import tensorutils as tx
from . import fourierutils as fx
import pickle
import sys


def _fexpr_n(expr_arr, n, precision=7):
    Rshape = expr_arr.shape
    expr_arrf = expr_arr.flatten()
    h = np.zeros(len(expr_arrf), dtype=object)
    for i in range(len(expr_arrf)):
        f_re = sp.lambdify(phi, 1/2/sp.pi*expr_arrf[i]*sp.cos(-1*(n*phi)))
        f_im = sp.lambdify(phi, 1/2/sp.pi*expr_arrf[i]*sp.sin(-1*(n*phi)))
        t_re,_ = quad(f_re, 0, 2*np.pi)
        t_im,_ = quad(f_im, 0, 2*np.pi)
        h[i] = round(t_re, precision)+round(t_im, precision)*1j
    h = h.reshape(Rshape)
    return h


def _convolve_ftensors(nR1, nR2, M=16, dtype=object):
    test_prod = tx.tensor_product(nR1[0], nR2[0])
    ans = np.zeros(dtype=dtype, shape=(2*M+1,)+test_prod.shape)
    for n in np.arange(-M, M+1):
        for m in np.arange(-M, M+1):
            try:
                ans[n2i(n, M)] += tx.tensor_product(nR1[n2i(m, M)], nR2[n2i(n-m, M)])
            except IndexError:
                pass
    return ans


def load_pickle(filename):
    return np.load(filename, allow_pickle=True)


def save_fform_dict(filename, fform_dict):
    pickle.dump(fform_dict, open(filename, 'wb'))


def load_fform_dict(filename):
    return pickle.load(open(filename, 'rb'))


def generate_uncontracted_fourier_transforms(aoi, verbose=True, filename_prefix='h7', M=16):

    include_quadrupole = True

    ## 
    ## First define all of the types of 
    ## tensors we will need to form the
    ## tensor product.
    ## 
    F = np.array([Fx, Fy, Fz])
    np.set_printoptions(threshold=sys.maxsize)
    R = np.array(util.rotation_matrix3symb([0, 0, 1], phi, ndigits=5))
    Id = np.identity(3)
    k_out = fx.substitute_into_array(np.array([-sp.sin(theta), 0, -sp.cos(theta)]), (theta, aoi))
    k_in = fx.substitute_into_array(np.array([-sp.sin(theta), 0, sp.cos(theta)]), (theta, aoi))
    proj = Id - tx.tensor_product(k_out, k_out)
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
    rproj[fx.n2i(0, M)] = proj
    rproj_x = np.zeros(shape=(2*M+1,)+proj_x.shape, dtype=object)
    rproj_x[fx.n2i(0, M)] = proj_x
    rproj_y = np.zeros(shape=(2*M+1,)+proj_y.shape, dtype=object)
    rproj_y[fx.n2i(0, M)] = proj_y
    rproj_z = np.zeros(shape=(2*M+1,)+proj_z.shape, dtype=object)
    rproj_z[fx.n2i(0, M)] = proj_z
    rR = [_fexpr_n(R, m) for m in np.arange(-M, M+1)]
    rF = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rF[fx.n2i(0, M)] = F
    rFc = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rFc[fx.n2i(0, M)] = sp.conjugate(F)
    r1_x = _convolve_ftensors(rproj_x, rR)
    r1_y = _convolve_ftensors(rproj_y, rR)
    r1_z = _convolve_ftensors(rproj_z, rR)
    rk_out = np.zeros(shape=(2*M+1,)+k_out.shape, dtype=object)
    rk_out[fx.n2i(0, M)] = k_out
    rk_in = np.zeros(shape=(2*M+1,)+k_in.shape, dtype=object)
    rk_in[fx.n2i(0, M)] = k_in

    ## 
    ## This is so I can address the P and
    ## S components individually.
    ## 
    r1_0 = np.zeros(shape=r1_x.shape, dtype=object)
    r1_s = np.array([np.array([r1_0[fx.n2i(m, M)], r1_y[fx.n2i(m, M)], r1_0[fx.n2i(m, M)]]) for m in np.arange(-M, M+1)])
    r1_p = np.array([np.array([r1_x[fx.n2i(m, M)], r1_0[fx.n2i(m, M)], r1_z[fx.n2i(m, M)]]) for m in np.arange(-M, M+1)])

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
        h1 = np.array([tx.tensor_contract(r1[fx.n2i(m, M)], [[1, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h1 done!')
        r2 = _convolve_ftensors(h1, h1)
        h2 = np.array([tx.tensor_contract(r2[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h2 done!')
        r3 = _convolve_ftensors(rR, rF)
        h3 = np.array([tx.tensor_contract(r3[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h3 done!')
        h4 = _convolve_ftensors(h2, h3)
        util.oprint(verbose, 'h4 done!')
        h5 = _convolve_ftensors(h4, h3)
        util.oprint(verbose, 'h5 done!')
        h6 = _convolve_ftensors(h5, h3)
        util.oprint(verbose, 'h6 done!')
        h7 = _convolve_ftensors(h6, h3)
        util.oprint(verbose, 'h7 done!')
        h7_arr_term1.append(h7)
        if include_quadrupole is True:
            r4 = _convolve_ftensors(rR, -1j*rk_in)
            h8 = np.array([tx.tensor_contract(r4[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
            util.oprint(verbose, 'h8 done!')
            h9 = _convolve_ftensors(h7, h8)
            util.oprint(verbose, 'h9 done!')
            h7_arr_term2.append(h9)
            h7_arr_term3.append(-1*h9)
            h10 = _convolve_ftensors(h9, -1*h8)
            util.oprint(verbose, 'h10 done!')
            h7_arr_term4.append(h10)
        
    ##
    ## Now I form four of these tensors, each
    ## pertaining to a particular polarization
    ## combination.
    ##
    list_of_terms = [h7_arr_term1, h7_arr_term2, h7_arr_term3, h7_arr_term4]
    nterms = len(list_of_terms)
    util.oprint(verbose, 'started substitution!')

    h7_pp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_pp[i] = fx.substitute_into_array(h7_arr[0], (Fx, np.cos(aoi)), (Fy, 0), (Fz, np.sin(aoi)))
    util.oprint(verbose, 'done 1!')

    h7_ps = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ps[i] = fx.substitute_into_array(h7_arr[1], (Fx, np.cos(aoi)), (Fy, 0), (Fz, np.sin(aoi)))
    util.oprint(verbose, 'done 2!')
    
    h7_sp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_sp[i] = fx.substitute_into_array(h7_arr[0], (Fx, 0), (Fy, -1), (Fz, 0))
    util.oprint(verbose, 'done 3!')

    h7_ss = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ss[i] = fx.substitute_into_array(h7_arr[1], (Fx, 0), (Fy, -1), (Fz, 0))
    util.oprint(verbose, 'done 4!')

    util.oprint(verbose, 'we did it!')
    

    ##
    ## Now to generate the fourier transformed
    ## SHG formulae for a particular tensor chi,
    ## all you need to do is to contract these
    ## tensors in a certain way with chi x chi.
    ##
    ## All that's left is to save these h7 tensors
    ## to a file.
    ##
    np.save(filename_prefix+'_pp', h7_pp)
    np.save(filename_prefix+'_ps', h7_ps)
    np.save(filename_prefix+'_sp', h7_sp)
    np.save(filename_prefix+'_ss', h7_ss)

def generate_uncontracted_fourier_transforms_symb(verbose=True, filename_prefix='h7', M=16):

    include_quadrupole=True

    ## 
    ## First define all of the types of 
    ## tensors we will need to form the
    ## tensor product.
    ## 
    F = np.array([Fx, Fy, Fz])
    np.set_printoptions(threshold=sys.maxsize)
    R = np.array(util.rotation_matrix3symb([0, 0, 1], phi, ndigits=5))
    Id = np.identity(3)
##     k_out = fx.substitute_into_array(np.array([-sp.sin(theta), 0, -sp.cos(theta)]), (theta, aoi))
##     k_in = fx.substitute_into_array(np.array([-sp.sin(theta), 0, sp.cos(theta)]), (theta, aoi))
    k_out = np.array([-sp.sin(theta), 0, -sp.cos(theta)])
    k_in = np.array([-sp.sin(theta), 0, sp.cos(theta)])
    proj = Id - tx.tensor_product(k_out, k_out)
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
    rproj[fx.n2i(0, M)] = proj
    rproj_x = np.zeros(shape=(2*M+1,)+proj_x.shape, dtype=object)
    rproj_x[fx.n2i(0, M)] = proj_x
    rproj_y = np.zeros(shape=(2*M+1,)+proj_y.shape, dtype=object)
    rproj_y[fx.n2i(0, M)] = proj_y
    rproj_z = np.zeros(shape=(2*M+1,)+proj_z.shape, dtype=object)
    rproj_z[fx.n2i(0, M)] = proj_z
    rR = [_fexpr_n(R, m) for m in np.arange(-M, M+1)]
    rF = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rF[fx.n2i(0, M)] = F
    rFc = np.zeros(shape=(2*M+1,)+F.shape, dtype=object)
    rFc[fx.n2i(0, M)] = sp.conjugate(F)
    r1_x = _convolve_ftensors(rproj_x, rR)
    r1_y = _convolve_ftensors(rproj_y, rR)
    r1_z = _convolve_ftensors(rproj_z, rR)
    rk_out = np.zeros(shape=(2*M+1,)+k_out.shape, dtype=object)
    rk_out[fx.n2i(0, M)] = k_out
    rk_in = np.zeros(shape=(2*M+1,)+k_in.shape, dtype=object)
    rk_in[fx.n2i(0, M)] = k_in

    ## 
    ## This is so I can address the P and
    ## S components individually.
    ## 
    r1_0 = np.zeros(shape=r1_x.shape, dtype=object)
    r1_s = np.array([np.array([r1_0[fx.n2i(m, M)], r1_y[fx.n2i(m, M)], r1_0[fx.n2i(m, M)]]) for m in np.arange(-M, M+1)])
    r1_p = np.array([np.array([r1_x[fx.n2i(m, M)], r1_0[fx.n2i(m, M)], r1_z[fx.n2i(m, M)]]) for m in np.arange(-M, M+1)])

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
        h1 = np.array([tx.tensor_contract(r1[fx.n2i(m, M)], [[1, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h1 done!')
        r2 = _convolve_ftensors(h1, h1)
        h2 = np.array([tx.tensor_contract(r2[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h2 done!')
        r3 = _convolve_ftensors(rR, rF)
        h3 = np.array([tx.tensor_contract(r3[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
        util.oprint(verbose, 'h3 done!')
        h4 = _convolve_ftensors(h2, h3)
        util.oprint(verbose, 'h4 done!')
        h5 = _convolve_ftensors(h4, h3)
        util.oprint(verbose, 'h5 done!')
        h6 = _convolve_ftensors(h5, h3)
        util.oprint(verbose, 'h6 done!')
        h7 = _convolve_ftensors(h6, h3)
        util.oprint(verbose, 'h7 done!')
        h7_arr_term1.append(h7)
        if include_quadrupole is True:
            r4 = _convolve_ftensors(rR, -1j*rk_in)
            h8 = np.array([tx.tensor_contract(r4[fx.n2i(m, M)], [[0, 2]]) for m in np.arange(-M, M+1)])
            util.oprint(verbose, 'h8 done!')
            h9 = _convolve_ftensors(h7, h8)
            util.oprint(verbose, 'h9 done!')
            h7_arr_term2.append(h9)
            h7_arr_term3.append(-1*h9)
            h10 = _convolve_ftensors(h9, -1*h8)
            util.oprint(verbose, 'h10 done!')
            h7_arr_term4.append(h10)
        
    ##
    ## Now I form four of these tensors, each
    ## pertaining to a particular polarization
    ## combination.
    ##
    list_of_terms = [h7_arr_term1, h7_arr_term2, h7_arr_term3, h7_arr_term4]
    nterms = len(list_of_terms)
    util.oprint(verbose, 'started substitution!')

    h7_pp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_pp[i] = fx.substitute_into_array(h7_arr[0], (Fx, sp.cos(theta)), (Fy, 0), (Fz, sp.sin(theta)))
    util.oprint(verbose, 'done 1!')

    h7_ps = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ps[i] = fx.substitute_into_array(h7_arr[1], (Fx, sp.cos(theta)), (Fy, 0), (Fz, sp.sin(theta)))
    util.oprint(verbose, 'done 2!')
    
    h7_sp = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_sp[i] = fx.substitute_into_array(h7_arr[0], (Fx, 0), (Fy, -1), (Fz, 0))
    util.oprint(verbose, 'done 3!')

    h7_ss = np.empty(nterms, dtype=object)
    for i,h7_arr in enumerate(list_of_terms):
        h7_ss[i] = fx.substitute_into_array(h7_arr[1], (Fx, 0), (Fy, -1), (Fz, 0))
    util.oprint(verbose, 'done 4!')

    util.oprint(verbose, 'we did it!')
    

    ##
    ## Now to generate the fourier transformed
    ## SHG formulae for a particular tensor chi,
    ## all you need to do is to contract these
    ## tensors in a certain way with chi x chi.
    ##
    ## All that's left is to save these h7 tensors
    ## to a file.
    ##
    np.save(filename_prefix+'_pp', h7_pp)
    np.save(filename_prefix+'_ps', h7_ps)
    np.save(filename_prefix+'_sp', h7_sp)
    np.save(filename_prefix+'_ss', h7_ss)


def generate_contracted_fourier_transforms(filename_prefix, chi_dipole, chi_quadrupole, M=16, ndigits=None, verbose=True):

    ##
    ## First we load the uncontracted fourier
    ## transforms which we generated with
    ## generate_uncontracted_fourier_transforms.
    ## Each of the following arrays holds four
    ## items (with various shapes).
    ##
    h7_pp_terms = load_pickle(filename_prefix+'_pp.npy') 
    h7_ps_terms = load_pickle(filename_prefix+'_ps.npy') 
    h7_sp_terms = load_pickle(filename_prefix+'_sp.npy') 
    h7_ss_terms = load_pickle(filename_prefix+'_ss.npy') 

    ##
    ## Now I build a useful set of arrays
    ## which we will use later to reduce
    ## the number of lines in this program
    ##
    pcs = ['PP', 'PS', 'SP', 'SS']
    terms_dict = {pc:load_pickle(filename_prefix+'_'+pc.lower()+'.npy') for pc in pcs}

    contraction_lists_1 = [[[0, 6], [2, 7], [3, 8]],
                           [[0, 7], [2, 8], [3, 9]],
                           [[0, 7], [2, 9], [3, 10], [6, 8]],
                           [[0, 8], [2, 10], [3, 11], [6, 9]]]
    contraction_lists_2 = [[[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]],
                           [[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]]]

    chi_list_1 = [chi_dipole, chi_dipole, chi_quadrupole, chi_quadrupole]
    chi_list_2 = [chi_dipole, chi_quadrupole, chi_dipole, chi_quadrupole]

    util.oprint(verbose, 'finished preparation!')

    ##
    ## Now we do the contraction with chi x chi,
    ## for each polarization combination and for
    ## each term. Each of the tensors h7_pp, ...
    ## has 4 elements, correponsing to the four
    ## terms in |P+ikQ|^2.
    ##
    fform_dict = {}
    for pc,h7_pc in terms_dict.items():
        fform_dict[pc] = np.zeros(shape=(2*M+1,), dtype=object)
        for term in range(len(h7_pc)):
            t8_pc_term = np.array([tx.tensor_contract(tx.tensor_product(h7_pc[term][fx.n2i(m, M)], chi_list_1[term]), contraction_lists_1[term]) for m in np.arange(-M, M+1)])
            t9_pc_term = np.array([tx.tensor_contract(tx.tensor_product(t8_pc_term[fx.n2i(m, M)], chi_list_2[term]), contraction_lists_2[term]) for m in np.arange(-M, M+1)])
            fform_dict[pc]+=np.copy(t9_pc_term)
            util.oprint(verbose, 'finished term %s!' % term)
        if ndigits is not None:
            fform_dict[pc] = util.round_complex_tensor(fform_dict[pc], ndigits)
        util.oprint(verbose, 'finished %s!' % pc)

    return fform_dict

def generate_contracted_fourier_transforms_complex(filename_prefix, chi_dipole, chi_quadrupole, M=16, ndigits=None, verbose=True):

    ##
    ## First we check if all the parameters in 
    ## chi_dipole and chi_quadrupole are real.
    ## If not, raise an error.
    ##
    for chi in [chi_dipole, chi_quadrupole]:
        free_symbols = util.free_symbols_of_array(chi)
        for fs in free_symbols:
            if fs.is_real is not True:
                raise ValueError('Parameters of chi must all be real: %s' % str(fs))

    ##
    ## First we load the uncontracted fourier
    ## transforms which we generated with
    ## generate_uncontracted_fourier_transforms.
    ## Each of the following arrays holds four
    ## items (with various shapes).
    ##
    h7_pp_terms = load_pickle(filename_prefix+'_pp.npy') 
    h7_ps_terms = load_pickle(filename_prefix+'_ps.npy') 
    h7_sp_terms = load_pickle(filename_prefix+'_sp.npy') 
    h7_ss_terms = load_pickle(filename_prefix+'_ss.npy') 

    ##
    ## Now I build a useful set of arrays
    ## which we will use later to reduce
    ## the number of lines in this program
    ##
    pcs = ['PP', 'PS', 'SP', 'SS']
    terms_dict = {pc:load_pickle(filename_prefix+'_'+pc.lower()+'.npy') for pc in pcs}

    contraction_lists_1 = [[[0, 6], [2, 7], [3, 8]],
                           [[0, 7], [2, 8], [3, 9]],
                           [[0, 7], [2, 9], [3, 10], [6, 8]],
                           [[0, 8], [2, 10], [3, 11], [6, 9]]]
    contraction_lists_2 = [[[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]],
                           [[0, 3], [1, 4], [2, 5]],
                           [[0, 4], [1, 6], [2, 7], [3, 5]]]

    chi_list_1 = [chi_dipole, chi_dipole, chi_quadrupole, chi_quadrupole]
    chi_list_2 = [util.conjugate_tensor(chi_dipole), util.conjugate_tensor(chi_quadrupole), util.conjugate_tensor(chi_dipole), util.conjugate_tensor(chi_quadrupole)]

    util.oprint(verbose, 'finished preparation!')

    ##
    ## Now we do the contraction with chi x chi,
    ## for each polarization combination and for
    ## each term. Each of the tensors h7_pp, ...
    ## has 4 elements, correponsing to the four
    ## terms in |P+ikQ|^2.
    ##
    fform_dict = {}
    for pc,h7_pc in terms_dict.items():
        fform_dict[pc] = np.zeros(shape=(2*M+1,), dtype=object)
        for term in range(len(h7_pc)):
            t8_pc_term = np.array([tx.tensor_contract(tx.tensor_product(h7_pc[term][fx.n2i(m, M)], chi_list_1[term]), contraction_lists_1[term]) for m in np.arange(-M, M+1)])
            t9_pc_term = np.array([tx.tensor_contract(tx.tensor_product(t8_pc_term[fx.n2i(m, M)], chi_list_2[term]), contraction_lists_2[term]) for m in np.arange(-M, M+1)])
            fform_dict[pc]+=np.copy(t9_pc_term)
            util.oprint(verbose, 'finished term %s!' % term)
        if ndigits is not None:
            fform_dict[pc] = util.round_complex_tensor(fform_dict[pc], ndigits)
        util.oprint(verbose, 'finished %s!' % pc)

    return fform_dict
