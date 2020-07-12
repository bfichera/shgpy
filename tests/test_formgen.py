import unittest

import random
import logging

import shgpy
import shgpy.tensor_definitions as td
from shgpy.formgen import (
    formgen_just_dipole,
    formgen_dipole_quadrupole,
)
import shgpy.shg_symbols as S
from shgpy.plotter import easy_plot
import shgpy.fformfit

show_plot = False

logging.getLogger(__name__)

MANUAL = False


class TestFormGen(unittest.TestCase):

    def test_formgen(self):
        t_dip = shgpy.make_tensor_real(td.dipole['T_d'])
        t_quad = shgpy.make_tensor_real(td.quadrupole['T_d'])
        form1 = formgen_just_dipole(t_dip, S.theta)
        form2 = formgen_dipole_quadrupole(t_dip, t_quad, S.theta)
        
        t_dip = shgpy.make_tensor_complex(td.dipole['T_d'])
        t_quad = shgpy.make_tensor_complex(td.quadrupole['T_d'])
        form3 = formgen_just_dipole(t_dip, S.theta)
        form4 = formgen_dipole_quadrupole(t_dip, t_quad, S.theta)
        forms = [form1, form2, form3, form4]
        for form in forms:
            form.apply_phase_shift(S.psi, S.phi)
            dat = shgpy.form_to_dat(
                form,
                {
                    k:random.uniform(-1, 1)
                    for k in form.get_free_symbols()
                    if k != S.phi
                },
                1000,
            )
            if MANUAL:
                easy_plot(
                    [dat],
                    [{'linestyle':'-', 'color':'blue'}],
                    dat.get_keys(),
                    show_plot=False,
                )
