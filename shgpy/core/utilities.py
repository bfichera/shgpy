import numpy as np
import sympy as sp
from sympy.solvers import solve
import itertools
from warnings import warn
import logging

logging.getLogger(__name__)


def particularize(tensor, exclude=[]):

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


def _make_parameters_in_expr_complex(expr, prefix=('real_', 'imag_'), suffix=('', '')):
    free_symbols = expr.free_symbols
    for fs in free_symbols:
        if fs.is_real is None:
            expr = expr.subs(str(fs), sp.symbols(prefix[0]+str(fs)+suffix[0], real=True)+sp.I*sp.symbols(prefix[1]+str(fs)+suffix[1], real=True))
    return expr


def make_tensor_complex(tensor, prefix=('real_', 'imag_'), suffix=('', '')):
    shape = tensor.shape
    tensor = tensor.flatten()
    for i in range(len(tensor)):
        tensor[i] = _make_parameters_in_expr_complex(sp.sympify(tensor[i]), prefix=prefix, suffix=suffix)
    return np.reshape(tensor, shape)


def rotation_matrix3(n, t):
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
