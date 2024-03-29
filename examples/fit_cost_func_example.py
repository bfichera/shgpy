from pathlib import Path

import numpy as np
import lmfit

from shgpy.fformfit import load_func
from shgpy.plotter import easy_plot
import shgpy
from shgpy import shg_symbols as S


# Define filepaths and constants
ROOT_DIR = Path(__file__).parent
COST_FUNC_FILENAME = ROOT_DIR / 'func' / 'Td.cf'
FFORM_FILENAME = ROOT_DIR / 'fform' / 'T_d-S_2-S_2(110)-particularized-byform.p'
DATA_DIR = ROOT_DIR / 'Data'
FIT_KWARGS_DUAL_ANNEALING = {
    'max_nfev':30000,
    'no_local_search':True,
}
METHOD = 'dual_annealing'
MIN = -100
MAX = 100


data_filenames_dict_1 = {
    'PP':DATA_DIR / 'dataPP.csv',
    'PS':DATA_DIR / 'dataPS.csv',
    'SP':DATA_DIR / 'dataSP.csv',
    'SS':DATA_DIR / 'dataSS.csv',
} 


# Load fform. Once the cost function has been generated, this is not
# strictly necessary.
fform = shgpy.load_fform(FFORM_FILENAME)
# Apply arbitrary phase shift to fform.
fform.apply_phase_shift(S.psi)

print(fform.get_free_symbols())

# Initialize lmfit parameters object
params = lmfit.Parameters()
for k_fs in fform.get_free_symbols():
    _min = MIN
    _max = MAX
    params.add(
        str(k_fs),
        value=np.random.uniform(_min, _max),
        min=_min,
        max=_max,
    )
params.pretty_print()


# Load cost func (generated by examples/gen_cost_func_example.py)
cost_func_unsanitized = load_func(COST_FUNC_FILENAME)
    

# Sanitize cost func inputs to be compatible with lmfit.minimize
def cost_func(params):
    pars = [params[k].value for k in params.keys()]
    x = np.array(pars).astype(float)
    return cost_func_unsanitized(x)


# Load the data
dat = shgpy.load_data(
    data_filenames_dict_1,
    'degrees',
)
fdat = shgpy.dat_to_fdat(dat)

# Minimize the cost function
ret = lmfit.minimize(
    cost_func,
    params,
    method=METHOD,
    **FIT_KWARGS_DUAL_ANNEALING,
)

# Plot the result
subs_dict = {}
for k_fs in fform.get_free_symbols():
    subs_dict[k_fs] = ret.params[str(k_fs)].value

print(lmfit.fit_report(ret))
print('OPTIMAL FUNCTION VALUE:', ret.residual)

fit_dat = shgpy.fform_to_dat(fform, subs_dict, 1000)
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
    pcs_to_include=['PP', 'PS', 'SP', 'SS'],
    show_plot=True,
    filename=None,
    show_legend=False,
)
