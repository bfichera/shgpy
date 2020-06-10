import numpy as np
import logging

import shgpy
import shgpy.shg_symbols as S
from shgpy.plotter import easy_plot
from shgpy.fformfit import least_squares_fit_with_bounds

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
}
dat, fdat = shgpy.load_data_and_fourier_transform(data_filenames_dict, 'degrees')

fform_filename = 'fform/T_d-S_2-S_2(110)-particularized.p'

fform = shgpy.load_fform(fform_filename)
fform.apply_phase_shift(S.psi)

iterable = {}
for k,v in fform.get_items():
    if k in ['PP', 'PS']:
        iterable[k] = v
fform = shgpy.fFormContainer(iterable)

guess_dict = {}
for fs in fform.get_free_symbols():
    guess_dict[fs] = 1

bounds_dict = {}
for fs in fform.get_free_symbols():
    if fs == S.psi:
        bounds_dict[fs] = (-np.pi, np.pi)
    else:
        bounds_dict[fs] = (-2, 2)

ret = least_squares_fit_with_bounds(fform, fdat, guess_dict, bounds_dict)

fit_dat = shgpy.fform_to_dat(fform, ret.xdict, 1000)
easy_plot(
    list_of_dats=[dat, fit_dat],
    list_of_param_dicts=[
        {
            'linestyle':' ',
            'markerfacecolor':'none',
            'color':'blue',
            'marker':'o',
        },
        {
            'linestyle':'-',
            'color':'blue'
        },
    ],
    pcs_to_include=['PP', 'PS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)
