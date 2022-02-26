import logging
import numpy as np
import sympy as sp
from time import time

import shgpy
import shgpy.tensor_definitions as td
from shgpy.formgen import formgen

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

start = time()
form = formgen(sp.pi/18, t_eee=t_dipole)
mylogger.debug(f'Finished form generation. It took {time()-start} seconds.')

start = time()
fform = shgpy.form_to_fform(form)
mylogger.debug(f'Finished fform generation. It took {time()-start} seconds.')

save_filename = 'fform/Td.p'
shgpy.save_fform(fform, save_filename)
