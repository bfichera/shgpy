from pathlib import Path

import numpy as np
import lmfit

from shgpy.fformfit import load_model_func
from shgpy.plotter import easy_plot
import shgpy


ROOT_DIR = Path(__file__).parent
SAVE_FOLDER = ROOT_DIR / 'func' / 'Td'
FFORM_FILENAME = ROOT_DIR / 'fform' / 'T_d-S_2-S_2(110)-particularized-byform.p'
DATA_DIR = ROOT_DIR / 'Data'
FIT_KWARGS_DUAL_ANNEALING = {
    'max_nfev':30000,
    'no_local_search':True,
#    'iter_cb':lambda params, iter, resid, *args, **kws: print(f'n={iter}: resid={resid}'),
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


fform = shgpy.load_fform(FFORM_FILENAME)

print(fform.get_free_symbols())

model_func = load_model_func(fform, SAVE_FOLDER)

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
params.add(
    'psi',
    value=np.random.uniform(0.0, 2*np.pi),
    min=0.0,
    max=2*np.pi,
)
params.pretty_print()
    

def model(params, pc, m):
    pars = [params[k].value for k in params.keys()]
    psi = params['psi']
    x = np.array(pars).astype(float)
    return model_func(x, pc, m)*np.exp(-1j*m*psi)


def cost_func(params, df):
    total = 0
    for (pc, m), v in zip(df.index.to_numpy(), df.to_numpy()):
        ans = model(params, pc, m) - v[0]
        total += np.real(ans*np.conj(ans))
    return total


dat = shgpy.load_data(
    data_filenames_dict_1,
    'degrees',
)
fdat = shgpy.dat_to_fdat(dat)

ret_SA = lmfit.minimize(
    cost_func,
    params,
    method=METHOD,
    kws={
        'df':fdat.as_pandas(),
    },
    **FIT_KWARGS_DUAL_ANNEALING,
)
ret_fake = lmfit.minimize(
    cost_func,
    ret_SA.params,
    method='powell',
    kws={
        'df':fdat.as_pandas(),
    },
)
ret = ret_SA

print(lmfit.fit_report(ret))
print('OPTIMAL FUNCTION VALUE:', ret.residual)

fit_iterable = {
    pc:np.array([model(ret.params, pc, m) for m in np.arange(-16, 17)])
    for pc in ['PP', 'PS', 'SP', 'SS']
}
fit_fdat = shgpy.fDataContainer(fit_iterable, M=16)
fit_dat = shgpy.fdat_to_dat(fit_fdat, 1000)

easy_plot(
    [dat, fit_dat],
    [{}, {}],
    ['PP', 'PS', 'SP', 'SS'],
    show_plot=True,
)
