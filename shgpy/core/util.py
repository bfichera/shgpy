import numpy as np
import sympy as sp
from sympy.solvers import solve
import itertools
from warnings import warn


def particularize(tensor, exclude=[]):

    def swap_last_two_indices(tensor):
        rank = len(tensor.shape)
        dim = len(tensor[0])
        ans = np.zeros(shape=tensor.shape, dtype=tensor.dtype)
        for i in itertools.product(*[range(dim) for s in range(rank-2)]):
            for j in itertools.combinations_with_replacement(range(3), 2):
                ans[i][j] = tensor[i][(j[1], j[0])]
                ans[i][(j[1], j[0])] = tensor[i][j]
        return ans

    def get_sols_from_particularization(tensor, exclude=[]):
        tensor2 = swap_last_two_indices(tensor)
        sols = solve((tensor-tensor2).flatten(), dict=True, exclude=exclude)
        return sols

    def apply_sol_to_tensor(tensor, sol):
        return np.array([fa.subs(sol) for fa in tensor.flatten()]).reshape(tensor.shape)

    return apply_sol_to_tensor(tensor, get_sols_from_particularization(tensor, exclude=exclude)[0])


def free_symbols_of_array(array):
    total = []
    for a in array.flatten():
        total = total+list(sp.sympify(a).free_symbols)
    return set(total)


def make_parameters_complex(expr, prefix=('real_', 'imag_'), suffix=('', '')):
    free_symbols = expr.free_symbols
    for fs in free_symbols:
        if fs.is_real is None:
            expr = expr.subs(str(fs), sp.symbols(prefix[0]+str(fs)+suffix[0], real=True)+sp.I*sp.symbols(prefix[1]+str(fs)+suffix[1], real=True))
    return expr


def conjugate_tensor(tensor):
    ans = np.zeros(len(tensor.flatten()), dtype=object)
    for i,expr in enumerate(tensor.flatten()):
        ans[i] = sp.conjugate(sp.sympify(expr))
    return ans.reshape(tensor.shape)


def make_tensor_complex(tensor, prefix=('real_', 'imag_'), suffix=('', '')):
    shape = tensor.shape
    tensor = tensor.flatten()
    for i in range(len(tensor)):
        tensor[i] = make_parameters_complex(sp.sympify(tensor[i]), prefix=prefix, suffix=suffix)
    return np.reshape(tensor, shape)


def d2chi(d):
    ans = np.zeros(shape=(3,3,3), dtype=object)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if j == k:
                    ans[i][j][k] = d[i][j]
                else:
                    ans[i][j][k] = d[i][6-(j+k)]
    return ans


def gradient(expr, free_symbols):
    return np.array([sp.diff(expr, free_symbol) for free_symbol in free_symbols])


def round_float_tensor(t, ndigits):
    ans = t.flatten()
    for i in range(len(ans)):
        ans[i] = round(ans[i], ndigits)
    ans = ans.reshape(t.shape)
    return ans


def round_complex(z, ndigits):
    return round(np.real(z), ndigits)+1j*round(np.imag(z), ndigits)


def round_expr(expr, ndigits):
    try:
        return expr.xreplace({n:(round(sp.re(n), ndigits)+1j*round(sp.im(n), ndigits)) for n in expr.atoms(sp.Number)})
    except AttributeError:
        return round_complex(expr, ndigits)


def round_complex_tensor(t, ndigits):
    ans = t.flatten()
    for i in range(len(ans)):
        try:
            ans[i] = round_expr(ans[i], ndigits)
        except AttributeError:
            ans[i] = round(sp.re(ans[i]), ndigits)+1j*round(sp.im(ans[i]), ndigits)
    ans = ans.reshape(t.shape)
    return ans


def modsquared(expr):
    return expr*np.conjugate(expr)


def oprint(verbose, *items, filename=None, mode='a'):
    if verbose:
        print(*items)
    if filename is not None:
        f = open(filename, 'a')
        print(*items, file=f)
        f.close()


def rotation_matrix3(n, t):
    n_mag = norm(n)
    for i in range(3):
        n[i] /= n_mag
    c = np.cos(t)
    s = np.sin(t)
    nx,ny,nz = n
    row1 = [c+nx**2*(1-c), nx*ny*(1-c)-nz*s, nx*nz*(1-c) + ny*s]
    row2 = [ny*nx*(1-c)+nz*s, c+ny**2*(1-c), ny*nz*(1-c)-nx*s]
    row3 = [nz*nx*(1-c) - ny*s, nz*ny*(1-c) + nx*s, c+nz**2*(1-c)]
    return [row1, row2, row3]


def rotation_matrix3symb(n,t, ndigits=16):
    n_mag = normsymb(n)
    for i in range(3):
        n[i] /= n_mag
    c = sp.cos(t)
    s = sp.sin(t)
    nx,ny,nz = n
    row1 = [c+nx**2*(1-c), nx*ny*(1-c)-nz*s, nx*nz*(1-c) + ny*s]
    row2 = [ny*nx*(1-c)+nz*s, c+ny**2*(1-c), ny*nz*(1-c)-nx*s]
    row3 = [nz*nx*(1-c) - ny*s, nz*ny*(1-c) + nx*s, c+nz**2*(1-c)]
    ans = [row1, row2, row3]
    return ans


def normalize(data, val):
    ans = [[],[]]
    ans[0] = data[0]
    for i in range(len(data[1])):
        ans[1].append(data[1][i]/val)
    return ans


def levi_civita(i, j, k, first_index=0):
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


def cross_product(v1, v2):
    ans = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ans[i] += levi_civita(i, j, k)*v1[j]*v2[k]
    return ans


def norm(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normsymb(v):
    return sp.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def rotation_matrix_from_two_vectors(v_initial, v_final, accuracy=.01, ndigits=16):
    iden = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mag_v_initial = norm(v_initial)
    mag_v_final = norm(v_final)
    for i in range(3):
        v_initial[i] /= mag_v_initial
        v_final[i] /= mag_v_final
    cos_theta = sum([v_initial[i]*v_final[i] for i in range(3)])
    mag_theta = np.arccos(cos_theta)
    axis = cross_product(v_initial, v_final)
    done = False
    if axis == [0, 0, 0]:
        if [-v_initial[i] for i in range(3)] == v_final:
            ans = [[-iden[i][j] for i in range(3)] for j in range(3)]
            done = True
        elif [v_initial[i] for i in range(3)] == v_final:
            ans = iden
            done = True
    if not done:
        ans = rotation_matrix3(axis, mag_theta)
        mag_axis = norm(axis)
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
    return ans


