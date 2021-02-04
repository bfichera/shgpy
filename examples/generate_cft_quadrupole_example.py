import logging
import numpy as np
import sympy as sp
import time

import shgpy
import shgpy.fformgen
import shgpy.tensor_definitions as td

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

t_dipole = np.zeros(shape=(3,3,3), dtype=sp.Expr)

t_quad = shgpy.particularize(td.quadrupole['D_6h'])
t_quad = shgpy.make_tensor_real(t_quad)

save_filename = 'fform/None-None-D_6h(001)-particularized.p'

start = time.time()
shgpy.fformgen.generate_contracted_fourier_transforms(save_filename, 'uft/ufttheta', t_dipole, t_quad, ndigits=9)
mylogger.debug(f'Finished CFT generation. Took {time.time()-start} seconds.')
