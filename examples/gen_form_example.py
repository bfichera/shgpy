import logging
import numpy as np
import sympy as sp
import random

import shgpy
import shgpy.tensor_definitions as td
from shgpy.formgen import formgen_just_dipole
from shgpy.plotter import easy_plot
import shgpy.shg_symbols as S

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

form = formgen_just_dipole(t_dipole, 0.1745)

subs_dict = {}
for fs in form.get_free_symbols():
    if fs != S.phi:
        subs_dict[fs] = random.uniform(-1, 1)
dat = shgpy.form_to_dat(form, subs_dict, 1000)
easy_plot([dat], [{'linestyle':'-', 'color':'blue'}], dat.get_keys(), show_plot=True)
