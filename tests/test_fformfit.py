import unittest

import logging

import shgpy
from shgpy.fformfit import (
    least_squares_fit,
    least_squares_fit_with_bounds,
    basinhopping_fit,
    basinhopping_fit_with_bounds,
    basinhopping_fit_jac,
    basinhopping_fit_jac_with_bounds,
    dual_annealing_fit_with_bounds,
    gen_cost_func,
)
import shgpy.fformfit
from shgpy.plotter import easy_plot, easy_polar_plot
import numpy as np
import shgpy.shg_symbols as S

logger = logging.getLogger(__name__)

MANUAL = False


class TestFit(unittest.TestCase):

    fform_filename = 'tests/fform/T_d-S_2-S_2(110)-particularized.p'
    fform = shgpy.load_fform(fform_filename)
    data_filenames_dict = {
        'PP':'tests/Data/dataPP.csv',
        'PS':'tests/Data/dataPS.csv',
        'SP':'tests/Data/dataSP.csv',
        'SS':'tests/Data/dataSS.csv',
    }
    dat, fdat = shgpy.load_data_and_fourier_transform(
        data_filenames_dict,
        'degrees',
    )
    fform.apply_phase_shift(S.psi)
    guess_dict = {k:1 for k in fform.get_free_symbols()}
    bounds_dict = {
        k:((-2, 2) if k != S.psi else (-np.pi, np.pi))
        for k in fform.get_free_symbols()
    }
    free_symbols = fform.get_free_symbols()

    def test_minimal(self):
        shgpy.fformfit._make_energy_func_wrapper(
            self.fform,
            self.fdat,
            self.free_symbols,
            False,
            'tests/Data/myfilename.so',
        )
        shgpy.fformfit._load_func('tests/Data/myfilename.so')
        shgpy.fformfit._make_energy_func_wrapper(
            self.fform,
            self.fdat,
            self.free_symbols,
            False, 
            'tests/Data/myfilename.so',
        )
        shgpy.fformfit._load_func('tests/Data/myfilename.so')

    def test_least_squares(self):
        ret1 = least_squares_fit(self.fform, self.fdat, self.guess_dict)
        ret2 = least_squares_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
        )
        for ret in [ret1, ret2]:
            self.assertAlmostEqual(abs(ret.xdict[S.psi]), 1.59, delta=0.1)
            self.assertAlmostEqual(
                abs(ret.xdict[shgpy.map_to_real(S.zyx)]),
                1.23,
                delta=0.1,
            )

        fit_dat = shgpy.fform_to_dat(self.fform, ret1.xdict, 1000)
        if MANUAL:
            easy_plot(
                [self.dat, fit_dat],
                [
                    {
                        'linestyle':' ',
                        'markerfacecolor':'none',
                        'color':'blue', 'marker':'o',
                    },
                    {
                        'linestyle':'-',
                        'color':'blue',
                    },
                ],
                ['PP', 'PS', 'SP', 'SS'],
                show_plot=True,
                filename=None,
                show_legend=False,
            )
            easy_polar_plot(
                [self.dat, fit_dat],
                [
                    {
                        'linestyle':' ',
                        'markerfacecolor':'none',
                        'color':'blue', 'marker':'o'
                    },
                    {
                        'linestyle':'-',
                        'color':'blue',
                    },
                ],
                ['PP', 'PS', 'SP', 'SS'],
                show_plot=True,
                filename=None,
                show_legend=False,
            )

    def test_basinhopping(self):
        niter = 100
        ret1 = basinhopping_fit(
            self.fform,
            self.fdat,
            self.guess_dict,
            niter,
        )
        ret2 = basinhopping_fit_jac(
            self.fform,
            self.fdat,
            self.guess_dict,
            niter,
        )
        ret3 = basinhopping_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            niter,
        )
        ret4 = basinhopping_fit_jac_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            niter,
        )
        for ret in [ret1, ret2, ret3, ret4]:
            self.assertAlmostEqual(abs(ret.xdict[S.psi]), 1.59, delta=0.1)
            self.assertAlmostEqual(
                abs(ret.xdict[shgpy.map_to_real(S.zyx)]),
                1.23,
                delta=0.1,
            )

    def test_annealing(self):
        maxiter = 100
        ret1 = dual_annealing_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            maxiter,
        )
        for ret in [ret1]:
            self.assertAlmostEqual(abs(ret.xdict[S.psi]), 1.59, delta=0.1)
            self.assertAlmostEqual(
                abs(ret.xdict[shgpy.map_to_real(S.zyx)]),
                1.23,
                delta=0.1,
            )

    def test_save(self):

        ret1 = dual_annealing_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            maxiter=100,
            save_cost_func_filename='tests/Data/costfunc.so',
        )

        ret2 = dual_annealing_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            maxiter=100,
            load_cost_func_filename='tests/Data/costfunc.so',
        )

        ret3 = basinhopping_fit_jac_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            100,
            save_cost_func_filename='tests/Data/costfunc.so',
            grad_save_cost_func_filename_prefix='tests/Data/grad_costfunc',
        )
        ret4 = basinhopping_fit_jac_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            100,
            load_cost_func_filename='tests/Data/costfunc.so',
            load_grad_cost_func_filename_prefix='tests/Data/grad_costfunc',
        )
        ret5 = basinhopping_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            100,
            save_cost_func_filename='tests/Data/costfunc.so',
            chunk_cost_func=True,
        )
        ret6 = basinhopping_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            100,
            load_cost_func_filename='tests/Data/costfunc.so',
            chunk_cost_func=True,
        )
        gen_cost_func(
            self.fform,
            self.fdat,
            save_filename='tests/Data/generated_costfunc.so',
        )
        ret7 = dual_annealing_fit_with_bounds(
            self.fform,
            self.fdat,
            self.guess_dict,
            self.bounds_dict,
            maxiter=100,
            load_cost_func_filename='tests/Data/generated_costfunc.so',
        )

        for ret in [ret1, ret2, ret3, ret4, ret5, ret6, ret7]:
            self.assertAlmostEqual(abs(ret.xdict[S.psi]), 1.59, delta=0.1)
            self.assertAlmostEqual(
                abs(ret.xdict[shgpy.map_to_real(S.zyx)]),
                1.23,
                delta=0.1,
            )
