import logging
import numpy as np

import shgpy

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

data_filenames_dict = {
    'PP':'Data/dataPP.csv',
    'PS':'Data/dataPS.csv',
    'SP':'Data/dataSP.csv',
    'SS':'Data/dataSS.csv',
}

dat = shgpy.load_data(data_filenames_dict, 'degrees')
dat_df = dat.as_pandas(requested_angle_units='degrees', index='none')

fdat = shgpy.dat_to_fdat(dat)
fdat_df = fdat.as_pandas(index='multi')

fform = shgpy.load_fform('fform/T_d-S_2-S_2(110)-particularized-byform.p')
fform_df = fform.as_pandas(index='none')

form = shgpy.fform_to_form(fform)
form_df = form.as_pandas(index='multi')
