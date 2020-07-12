import unittest

import numpy as np
import shgpy
import shgpy.fformgen
import shgpy.tensor_definitions as td
import shgpy.shg_symbols as S
import logging
import random
from shgpy.plotter import easy_plot

_logger = logging.getLogger(__name__)

MANUAL = False


class TestfFormGen(unittest.TestCase):

    def test_generate_uncontracted_fourier_transforms(self):
        shgpy.fformgen.generate_uncontracted_fourier_transforms(
            0.1745,
            'tests/fform/uft10deg',
        )
        shgpy.fformgen.generate_uncontracted_fourier_transforms_symb(
            'tests/fform/uft_theta',
        )
    
    def test_generate_contracted_fourier_transforms_complex(self):

        t_dip = shgpy.make_tensor_complex(
            shgpy.transform(
                shgpy.particularize(td.dipole['T_d']),
                shgpy.rotation_matrix_from_two_vectors(
                    np.array([1, 1, 0]),
                    np.array([0, 0, 1]),
                ),
            ),
        )
        t_quad = shgpy.make_tensor_complex(
            np.zeros(shape=(3,3,3,3),dtype=object),
        )
        save_filename = 'tests/fform/T_d-S_2-S_2(110)-particularized-complex.p'
        shgpy.fformgen.generate_contracted_fourier_transforms(
            save_filename,
            'tests/fform/uft10deg',
            t_dip,
            t_quad,
            ndigits=4,
        )
        fform = shgpy.load_fform(save_filename)
        fform.apply_phase_shift(S.psi)
        form = shgpy.fform_to_form(fform)
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
            )

    def test_generate_contracted_fourier_transforms(self):

        t_dip = shgpy.make_tensor_real(
            shgpy.transform(
                shgpy.particularize(td.dipole['T_d']),
                shgpy.rotation_matrix_from_two_vectors(
                    np.array([1, 1, 0]),
                    np.array([0, 0, 1]),
                ),
            ),
        )
        t_quad = shgpy.make_tensor_real(
            np.zeros(shape=(3,3,3,3), dtype=object),
        )
        save_filename = 'tests/fform/T_d-S_2-S_2(110)-particularized.p'
        shgpy.fformgen.generate_contracted_fourier_transforms(
            save_filename,
            'tests/fform/uft10deg',
            t_dip,
            t_quad,
            ndigits=4,
        )
        fform = shgpy.load_fform(save_filename)
        fform.apply_phase_shift(S.psi)
        form = shgpy.fform_to_form(fform)
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
            )
