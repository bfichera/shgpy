import logging
import numpy as np

import shgpy
from shgpy.plotter import easy_plot, easy_polar_plot

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
    'SP':'Data/dataSP.csv',
    'SS':'Data/dataSS.csv',
}

dat = shgpy.load_data(data_filenames_dict, 'degrees')

easy_plot(
    list_of_dats=[dat],
    list_of_param_dicts=[
        {
            'linestyle':'-',
            'color':'blue',
            'markerfacecolor':'none',
            'marker':'o',
        },
    ],
    pcs_to_include=['PP', 'PS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)

dat.scale_data(100)
dat.phase_shift_data(np.pi/2, 'radians')
easy_plot(
    list_of_dats=[dat],
    list_of_param_dicts=[
        {
            'linestyle':'-',
            'color':'blue',
            'markerfacecolor':'none',
            'marker':'o',
        },
    ],
    pcs_to_include=['PP', 'PS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)

easy_polar_plot(
    list_of_dats=[dat],
    list_of_param_dicts=[
        {
            'linestyle':'-',
            'color':'blue',
            'markerfacecolor':'none',
            'marker':'o',
        },
    ],
    pcs_to_include=['PP', 'PS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)
