import numpy as np
import sympy as sp
from sympy.solvers import solve
import itertools
from warnings import warn


def _particularize_last_two(tensor, exclude=[]):

    def _swap_last_two_indices(tensor):
        rank = len(tensor.shape)
        dim = len(tensor[0])
        ans = np.zeros(shape=tensor.shape, dtype=tensor.dtype)
        for i in itertools.product(*[range(dim) for s in range(rank-2)]):
            for j in itertools.combinations_with_replacement(range(3), 2):
                ans[i][j] = tensor[i][(j[1], j[0])]
                ans[i][(j[1], j[0])] = tensor[i][j]
        return ans

    def _get_sols_from_particularization(tensor, exclude=[]):
        tensor2 = _swap_last_two_indices(tensor)
        sols = solve((tensor-tensor2).flatten(), dict=True, exclude=exclude)
        return sols

    def _apply_sol_to_tensor(tensor, sol):
        return np.array([fa.subs(sol) for fa in tensor.flatten()]).reshape(tensor.shape)

    if np.array_equal(_swap_last_two_indices(tensor), tensor):
        return tensor

    return _apply_sol_to_tensor(tensor, _get_sols_from_particularization(tensor, exclude=exclude)[0])


def _particularize_all(tensor, exclude=[]):
    
    def _apply_sol_to_tensor(tensor, sol):
        return np.array([fa.subs(sol) for fa in tensor.flatten()]).reshape(tensor.shape)

    expr_list = []
    rank = len(tensor.shape)
    dim = len(tensor[0])

    for i in itertools.product(*[range(dim) for s in range(rank)]):
        for j in set(itertools.permutations(i)):
            expr_list.append(tensor[i]-tensor[j])

    sols = solve(expr_list, dict=True, exclude=exclude)
    
    return _apply_sol_to_tensor(tensor, sols[0])
    

def particularize(tensor, exclude=[], permute_all_indices=False):
    """Particularize `tensor` (e.g. enforce ``chi_ijk = chi_ikj``).

    Because (e.g.) ``P_i = chi_ijk E_j E_k``, ``chi_ijk`` needs to be symmetric
    in its last two indices. This symmetry is not documented in SHG tables,
    so it needs to be implemented manually. Functions using the ``solve``
    function of sympy.solvers.

    Parameters
    ----------
    tensor : ndarray
        Tensor to be particularized. 

    exclude : list of sympy.Symbol objects
        Variables to exlude from the solver. Defaults to ``[]``.

    permute_all_indices : bool, optional
        Whether to permute all indices, not just the last two. Useful for
        materials where Kleinman symmetry is preserved. Default is `False`.

    Returns
    -------
    particularized_tensor : ndarray

    """
    if permute_all_indices is False:
        return _particularize_last_two(tensor, exclude=exclude)
    elif permute_all_indices is True:
        return _particularize_all(tensor, exclude=exclude)


def _make_parameters_in_expr_complex(expr, prefix=('real_', 'imag_'), suffix=('', '')):
    free_symbols = expr.free_symbols
    for fs in free_symbols:
        if fs.is_real is None:
            expr = expr.subs(str(fs), sp.symbols(prefix[0]+str(fs)+suffix[0], real=True)+sp.I*sp.symbols(prefix[1]+str(fs)+suffix[1], real=True))
    return expr


def _make_parameters_in_expr_real(expr):
    free_symbols = expr.free_symbols
    for fs in free_symbols:
        if fs.is_real is None:
            expr = expr.subs(str(fs), sp.symbols(str(fs), real=True))
    return expr


def make_tensor_complex(tensor, prefix=('real_', 'imag_'), suffix=('', '')):
    """Substitute e.g. ``x`` in sympy expression with ``real_x+1j*imag_x``.

    In sympy, variables initalized by ``x = sympy.symbols('x')`` are by
    default assumed to be complex. In order to make this more explicit
    (e.g. for
    :func:`~shgpy.fformgen.generate_contracted_fourier_transforms`),
    we replace ``x`` by ``real_x + 1j*imag_x``.


    Parameters
    ----------
    tensor : ndarray
    prefix : tuple of str, optional
        The prefixes of the newly-created real and imaginary variables.
        Defaults to ``('real_', 'imag_')``.
    suffix : tuple of str, optional
        The prefixes of the newly-created real and imaginary variables.
        Defaults to ``('', '')``.

    Returns
    -------
    complex_tensor : ndarray
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    for i in range(len(tensor)):
        tensor[i] = _make_parameters_in_expr_complex(sp.sympify(tensor[i]), prefix=prefix, suffix=suffix)
    return np.reshape(tensor, shape)


def make_tensor_real(tensor):
    """Substitute e.g. ``x`` in sympy expression with its real counterpart.

    In sympy, variables initalized by ``x = sympy.symbols('x')`` are by
    default assumed to be complex. In order to make this more explicit
    (e.g. for
    :func:`~shgpy.fformgen.generate_contracted_fourier_transforms`),
    we replace ``sympy.Symbol('x')`` by ``sympy.Symbol('x', real=True)``.


    Parameters
    ----------
    tensor : ndarray

    Returns
    -------
    real_tensor : ndarray
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    for i in range(len(tensor)):
        tensor[i] = _make_parameters_in_expr_real(sp.sympify(tensor[i]))
    return np.reshape(tensor, shape)
    

def rotation_matrix3(n, t):
    """Returns the 3x3 matrix which rotates by `t` about `n`

    Parameters
    ----------
    n : ndarray
        Axis of rotation (ndim = 3).
    t : int, float, ...
        Angle (in radians) to rotate by.

    Returns
    -------
    rotation_matrix3 : ndarray of float
    """
    n_mag = _norm(n)
    for i in range(3):
        n[i] /= n_mag
    c = np.cos(t)
    s = np.sin(t)
    nx,ny,nz = n
    row1 = [c+nx**2*(1-c), nx*ny*(1-c)-nz*s, nx*nz*(1-c) + ny*s]
    row2 = [ny*nx*(1-c)+nz*s, c+ny**2*(1-c), ny*nz*(1-c)-nx*s]
    row3 = [nz*nx*(1-c) - ny*s, nz*ny*(1-c) + nx*s, c+nz**2*(1-c)]
    return np.array([row1, row2, row3])


def rotation_matrix3symb(n,t, ndigits=16):
    """Duplicate of :func:`~shgpy.core.utilities.rotation_matrix3`, but accepts sympy.Symbol angle argument.

    Parameters
    ----------
    n : ndarray
        Axis of rotation (ndim = 3).
    t : sympy.Symbol, ...
        Angle to rotate by.

    Returns
    -------
    rotation_matrix3symb : ndarray of sympy.Expr
    """
    n_mag = _normsymb(n)
    for i in range(3):
        n[i] /= n_mag
    c = sp.cos(t)
    s = sp.sin(t)
    nx,ny,nz = n
    row1 = [c+nx**2*(1-c), nx*ny*(1-c)-nz*s, nx*nz*(1-c) + ny*s]
    row2 = [ny*nx*(1-c)+nz*s, c+ny**2*(1-c), ny*nz*(1-c)-nx*s]
    row3 = [nz*nx*(1-c) - ny*s, nz*ny*(1-c) + nx*s, c+nz**2*(1-c)]
    ans = np.array([row1, row2, row3], dtype=object)
    return ans


def _levi_civita(i, j, k, first_index=0):
    i -= first_index
    j -= first_index
    k -= first_index
    ans = 0
    pos_args = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    neg_args = [[1, 0, 2], [0, 2, 1], [2, 1, 0]]
    if [i, j, k] in pos_args:
        ans = 1
    if [i, j, k] in neg_args:
        ans = -1
    return ans


def _cross_product(v1, v2):
    ans = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ans[i] += _levi_civita(i, j, k)*v1[j]*v2[k]
    return np.array(ans)


def _norm(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _normsymb(v):
    return sp.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def rotation_matrix_from_two_vectors(v_initial, v_final, accuracy=.01, ndigits=16):
    """Return rotation matrix which takes `v_initial` -> `v_final`.

    Parameters
    ----------
    v_initial : ndarray
    v_final : ndarray
    accuracy : float, optional
        Precision to which `v_final` is achieved. Defaults to `0.1`.
    ndigits : int, optional
        Number of digits to round off to. Defaults to `16`.

    Returns
    -------
    rotation_matrix3 : ndarray of float
    """
    v_initial = v_initial.astype(float)
    v_final = v_final.astype(float)
    iden = np.identity(3)
    mag_v_initial = _norm(v_initial)
    mag_v_final = _norm(v_final)
    for i in range(3):
        v_initial[i] /= mag_v_initial
        v_final[i] /= mag_v_final
    cos_theta = sum([v_initial[i]*v_final[i] for i in range(3)])
    mag_theta = np.arccos(cos_theta)
    axis = _cross_product(v_initial, v_final)
    done = False
    if np.array_equal(axis, np.zeros(3, dtype=axis.dtype)):
        if np.array_equal(-v_initial, v_final):
            ans = -iden
            done = True
        elif np.array_equal(v_initial, v_final):
            ans = iden
            done = True
    if not done:
        ans = rotation_matrix3(axis, mag_theta)
        mag_axis = _norm(axis)
        for i in range(3):
            axis[i] /= mag_axis
        check = [np.dot(ans, v_initial)[i] - v_final[i] < accuracy for i in range(3)]
        if False in check:
            ans = rotation_matrix3(axis, -mag_theta)
            check = [np.dot(ans, v_initial)[i] - v_final[i] < accuracy for i in range(3)]
            if False in check:
                warn('Rotation matrix from two vectors failed precision check.')

    for i in range(len(ans)):
        for j in range(len(ans[0])):
            ans[i][j] = round(ans[i][j], ndigits)
    return np.array(ans)


def union(*arrays):
    """Union of multiple arrays."""
    ans = []
    for y in arrays:
        for x in y:
            if x not in ans:
                ans.append(x)
    return ans


def tensor_product(*tensors):
    """Tensor product of multiple tensors."""
    if len(tensors) == 2:
        return np.tensordot(tensors[0], tensors[1], axes=0)
    else:
        return np.tensordot(tensors[0], tensor_product(*tensors[1:]), axes=0)


def tensor_contract(tensor, index_pairs):
    """Contract a tensor against to pairs of indices.

    Parameters
    ----------
    tensor : ndarray
    index_pairs : array_like of array_like of int
        List of pairs of indices, e.g. ``[[1, 3], [2, 4]]``.
    """
    ans = tensor
    index_idx = list(range(len(np.shape(tensor))))
    index_pairs = index_pairs[:]
    while len(index_pairs) > 0:
        old_index_pair = index_pairs.pop(0)
        new_index_pair = [index_idx.index(old_index_pair[i]) for i in range(len(old_index_pair))]
        ans = np.trace(ans, axis1=new_index_pair[0], axis2=new_index_pair[1])
        for i in range(len(old_index_pair)):
            index_idx.remove(old_index_pair[i])
    return ans
    

def transform(tensor, operation):
    """Transform a tensor by a given operation.

    Parameters
    ----------
    tensor : ndarray
    operation : ndarray
        Operation to transform `tensor` by. Should be rank 2.

    Returns
    -------
    transformed_tensor : ndarray
    """
    rank = len(tensor.shape)
    args = [operation]*rank
    args.append(tensor)
    return tensor_contract(tensor_product(*args), [[2*i+1, 2*rank+i] for i in range(rank)])


def _free_symbols_of_array(array):
    total = []
    for a in array.flatten():
        total = total+list(sp.sympify(a).free_symbols)
    return set(total)


def _assert_real_params(chi):
    free_symbols = _free_symbols_of_array(chi)
    for fs in free_symbols:
        if fs.is_real is not True:
            raise ValueError('Parameters of chi must all be real: %s. Use shgpy.make_tensor_real or shgpy.make_tensor_complex.' % str(fs))


def map_to_real(sym):
    """Map a sympy.Symbol to its real counterpart."""
    return sp.symbols(str(sym), real=True)


