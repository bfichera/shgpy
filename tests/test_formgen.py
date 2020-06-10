import unittest

import random
import logging
import numpy as np
import sympy as sp

import shgpy
import shgpy.tensor_definitions as td
from shgpy.formgen import (
    formgen_just_dipole_complex,
    formgen_just_dipole_real,
    formgen_dipole_quadrupole_complex,
    formgen_dipole_quadrupole_real,
)
import shgpy.shg_symbols as S
from shgpy.plotter import easy_plot
import shgpy.fformfit

show_plot = False

logging.getLogger(__name__)

MANUAL = False


class TestFormGen(unittest.TestCase):

    def test_formgen(self):
        t_dip = td.dipole['T_d']
        t_quad = td.quadrupole['T_d']
        form1 = formgen_just_dipole_real(t_dip, S.theta)
        form2 = formgen_dipole_quadrupole_real(t_dip, t_quad, S.theta)
        
        t_dip = shgpy.make_tensor_complex(td.dipole['T_d'])
        t_quad = shgpy.make_tensor_complex(td.quadrupole['T_d'])
        form3 = formgen_just_dipole_complex(t_dip, S.theta)
        form4 = formgen_dipole_quadrupole_complex(t_dip, t_quad, S.theta)
        forms = [form1, form2, form3, form4]
        for form in forms:
            form.apply_phase_shift(S.psi, S.phi)
            dat = shgpy.form_to_dat(form, {k:random.uniform(-1, 1) for k in form.get_free_symbols() if k != S.phi}, 1000)
            if MANUAL:
                easy_plot([dat], [{'linestyle':'-', 'color':'blue'}], dat.get_keys(), show_plot=False)

#     def test_formfit(self):
#         R = shgpy.rotation_matrix_from_two_vectors(np.array([1, 1, 0]), np.array([0, 0, 1]))
#         t_dip = shgpy.transform(shgpy.particularize(td.dipole['T_d']), R)
#         form = formgen_just_dipole_real(t_dip, 0.1745)
#         fform = shgpy.form_to_fform(form)
#         fform.apply_phase_shift(S.psi)
#         data_filenames_dict = {
#             'PP':'tests/Data/dataPP.csv',
#             'PS':'tests/Data/dataPS.csv',
#             'SP':'tests/Data/dataSP.csv',
#             'SS':'tests/Data/dataSS.csv',
#         }
#         dat, fdat = shgpy.load_data_and_fourier_transform(data_filenames_dict, 'degrees')
#         guess_dict = {k:1 for k in fform.get_free_symbols()}
#         ret = shgpy.fformfit.least_squares_fit(fform, fdat, guess_dict)
#         self.assertAlmostEqual(abs(ret.xdict[S.psi]), 1.59, delta=0.1)
#         self.assertAlmostEqual(abs(ret.xdict[S.zyx]), 1.23, delta=0.1)
#         fit_dat = shgpy.fform_to_dat(fform, ret.xdict, 1000)
#         if MANUAL:
#             easy_plot([dat, fit_dat], [{'linestyle':' ', 'markerfacecolor':'none', 'color':'blue', 'marker':'o'}, {'linestyle':'-', 'color':'blue'}], ['PP', 'PS', 'SP', 'SS'], show_plot=True, filename=None, show_legend=False)
