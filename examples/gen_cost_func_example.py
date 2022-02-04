import logging

import shgpy
import shgpy.shg_symbols as S
from shgpy.fformfit import gen_cost_func

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
    'SP':'Data/dataSP.csv',
    'SS':'Data/dataSS.csv',
}
dat, fdat = shgpy.load_data_and_fourier_transform(data_filenames_dict, 'degrees')

fform_filename = 'fform/T_d-S_2-S_2(110)-particularized-byform.p'

fform = shgpy.load_fform(fform_filename)
fform.apply_phase_shift(S.psi)

cost_func = gen_cost_func(fform, fdat, chunk=True, method='clang')