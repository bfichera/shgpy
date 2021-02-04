import logging
import numpy as np
import sympy as sp
import time

import shgpy
import shgpy.fformgen
import shgpy.tensor_definitions as td

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

R = shgpy.rotation_matrix_from_two_vectors(
    np.array([1, 1, 0]),
    np.array([0, 0, 1]),
)

t_dipole = shgpy.particularize(td.dipole['T_d'])
t_dipole = shgpy.make_tensor_real(t_dipole)
t_dipole = shgpy.transform(t_dipole, R)

t_quad = np.zeros(shape=(3,3,3,3), dtype=sp.Expr)

save_filename = 'fform/T_d-S_2-S_2(110)-particularized.p'

start = time.time()
shgpy.fformgen.generate_contracted_fourier_transforms(save_filename, 'uft/uft10deg', t_dipole, t_quad, ndigits=9)
mylogger.debug(f'Finished CFT generation. Took {time.time()-start} seconds.')
