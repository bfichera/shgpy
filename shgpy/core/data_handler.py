import numpy as np
import sympy as sp
import csv
import pickle
from scipy.interpolate import interp1d
import logging
from copy import deepcopy
from .. import shg_symbols as S

logging.getLogger(__name__)


class DataContainer:

    def __init__(self, iterable, input_angle_units):
        self._input_angle_units = input_angle_units.lower()
        self._check_angle_units(input_angle_units)
        self._check_defining_iterable(iterable)
        self._input_angle_units = input_angle_units
        self._scale = 1
        self._data_dict = dict(iterable)
        self._phase_shift = 0
        self._offset = 0
        if input_angle_units == 'degrees':
            self._convert_data_dict_to_radians()

    def _check_angle_units(self, angle_units):
        if angle_units not in ['radians', 'degrees']:
            raise ValueError('angle_units input must be one of \'radians\' or \'degrees\'.')

    def _check_defining_iterable(self, iterable):
        def raise_error():
            raise ValueError('Invalid Data input')
        try:
            bad_key = False in [type(k) == str for k in dict(iterable).keys()]
            bad_val = False in [type(v) == np.ndarray and v[0].shape == v[1].shape and v.dtype != object and v.ndim == 2 for v in dict(iterable).values()]
        except:
            raise_error()
        if bad_key or bad_val:
            raise_error()

    def _convert_data_dict_to_radians(self):
        self._data_dict = {k:np.array([np.deg2rad(v[0]), np.copy(v[1])]) for k,v in self._data_dict.items()}

    def _get_data_dict_degrees(self):
        return {k:np.array([np.rad2deg(v[0]), np.copy(v[1])]) for k,v in self._data_dict.items()}

    def scale_data(self, scale_factor):
        if scale_factor < 0:
            raise ValueError('scale factor cannot be negative.')
        self._data_dict = {k:np.array([v[0], scale_factor*v[1]]) for k,v in self._data_dict.items()}
        self._scale *= scale_factor

    def subtract_min(self):
        minval = self.get_minval()
        for k,v in self.get_keys():
            self._data_dict[k] = np.array([v[0], v[1]-minval])
        self._offset -= minval

    def get_maxval(self):
        maximum_value = -np.inf
        max_pc = None
        for k,v in self._data_dict.items():
            if max(v[1]) > maximum_value:
                maximum_value = max(v[1])
                max_pc = k
        return max_pc, maximum_value

    def get_pc_maxval(self, pc):
        return max(self._data_dict[pc][1])

    def get_minval(self):
        minimum_value = np.inf
        min_pc = None
        for k,v in self._data_dict.items():
            if min(v[1]) < minimum_value:
                minimum_value = min(v[1])
                min_pc = k
        return min_pc, minimum_value

    def get_pc_minval(self, pc):
        return min(self._data_dict[pc][1])

    def get_offset(self):
        return self._offset

    def normalize_data(self, desired_maximum=1):
        _, maximum_value = self.get_maxval()
        self.scale_data(desired_maximum / maximum_value)

    def phase_shift_data(self, angle, angle_units):
        angle_units = angle_units.lower()
        self._check_angle_units(angle_units)
        if angle_units == 'degrees':
            angle = np.deg2rad(angle)
        self._data_dict = {k:np.array([v[0]+angle, v[1]]) for k,v in self._data_dict.items()}
        self._phase_shift += angle

    def get_pc(self, pc, requested_angle_units):
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return np.copy(self._data_dict[pc])
        if requested_angle_units == 'degrees':
            return np.copy(self._get_data_dict_degrees()[pc])

    def get_keys(self):
        return list(self._data_dict.keys())

    def get_values(self, requested_angle_units):
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return list(self._data_dict.values())
        elif requested_angle_units == 'degrees':
            return list(self._get_data_dict_degrees().values())

    def get_items(self, requested_angle_units):
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return list(self._data_dict.items())
        elif requested_angle_units == 'degrees':
            return list(self._get_data_dict_degrees().items())

    def get_scale(self):
        return self._scale

    def get_phase_shift(self, requested_angle_units):
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return self._phase_shift
        elif requested_angle_units == 'degrees':
            return np.rad2deg(self._phase_shift)

    def get_input_angle_units(self):
        return self._input_angle_units


class fDataContainer:

    def __init__(self, iterable, M=16):
        self._M = M
        self._check_defining_iterable(iterable)
        self._fdata_dict = dict(iterable)
        self._scale = 1
        self._phase_shift = 0

    def _check_defining_iterable(self, iterable):
        def raise_error():
            raise ValueError('Invalid Data input')
        try:
            bad_key = False in [type(k) == str for k in dict(iterable).keys()]
            bad_val = False in [type(v) == np.ndarray and v.shape == (2*self._M+1,) and v.dtype in [np.complex64, np.complex128] and v.ndim == 1 for v in dict(iterable).values()]
        except:
            raise_error()
        if bad_key or bad_val:
            raise_error()

    def _check_angle_units(self, angle_units):
        if angle_units not in ['radians', 'degrees']:
            raise ValueError('angle_units input must be one of \'radians\' or \'degrees\'.')

    def get_keys(self):
        return list(self._fdata_dict.keys())

    def get_values(self):
        return list(self._fdata_dict.values())

    def get_items(self):
        return list(self._fdata_dict.items())

    def get_scale(self):
        return self._scale

    def get_M(self):
        return self._M

    def get_phase_shift(self, requested_angle_units):
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'degrees':
            return np.rad2deg(self._phase_shift)
        return self._phase_shift
        
    def scale_fdata(self, scale_factor):
        if scale_factor < 0:
            raise ValueError('scale factor cannot be negative.')
        self._fdata_dict = {k:scale_factor*v for k,v in self.get_items()}
        self._scale *= scale_factor

    def get_pc(self, pc):
        return np.copy(self._fdata_dict[pc])

    def get_maxval(self):
        maximum_value = -np.inf
        max_pc = None
        for k,v in self.get_items():
            this_max = max([abs(nv) for nv in v])
            if this_max > maximum_value:
                maximum_value = this_max
                max_pc = k
        return max_pc, maximum_value

    def get_pc_maxval(self, pc):
        v = self._fdata_dict[pc]
        maximum_value = max([abs(nv) for nv in v])
        return maximum_value
        
    def normalize_fdata(self, desired_maximum):
        _, maximum_value = self.get_maxval()
        self.scale_fdata(desired_maximum / maximum_value)

    def phase_shift_fdata(self, angle, angle_units):
        angle_units = angle_units.lower()
        self._check_angle_units(angle_units)
        if angle_units == 'degrees':
            angle = np.deg2rad(angle)
        for k in self.get_keys():
            fexpr = self._fdata_dict[k]
            ans = np.zeros(shape=fexpr.shape, dtype=fexpr.dtype)
            for m in np.arange(-self._M, self._M + 1):
                ans[n2i(m, self._M)] = fexpr[n2i(m, self._M)] * (np.cos(m * angle)+1j*np.sin(m * angle))
            self._fdata_dict[k] = ans
        self._phase_shift += angle


class fFormContainer:

    def __init__(self, iterable, M=16):
        self._M = M
        self._check_defining_iterable(iterable)
        self._fform_dict = dict(iterable)
        self._sympify()

    def _check_defining_iterable(self, iterable):
        def raise_error():
            raise ValueError('Invalid Data input.')
        try:
            bad_key = False in [type(k) == str for k in dict(iterable).keys()]
            bad_val = False in [type(v) == np.ndarray and v.shape == (2*self._M+1,) and v.dtype == object and v.ndim == 1 for v in dict(iterable).values()]
        except:
            raise_error()
        if bad_key:
            raise_error()
        if bad_val:
            raise_error()

    def _sympify(self):
        for k in self.get_keys():
            for m in np.arange(-self._M, self._M+1):
                self._fform_dict[k][n2i(m, self._M)] = sp.sympify(self._fform_dict[k][n2i(m, self._M)])

    def get_free_symbols(self):
        free_symbols = []
        for k in self.get_keys():
            for m in np.arange(-self._M, self._M+1):
                for fs in self._fform_dict[k][n2i(m, self._M)].free_symbols:
                    free_symbols.append(fs)
        free_symbols = list(set(free_symbols))
        free_symbols.sort(key=str)
        return free_symbols

    def subs(self, subs_array):
        subs_fform_dict = {}
        for k,v in self.get_items():
            subs_fform_dict[k] = np.zeros(shape=(2*self._M+1,), dtype=object)
            for m in np.arange(-self._M, self._M+1):
                subs_fform_dict[k][n2i(m, self._M)] = self._fform_dict[k][n2i(m, self._M)].subs(subs_array)
        self._fform_dict = subs_fform_dict

    def apply_phase_shift(self, phase_shift):
        new_fform_dict = {k:np.zeros(shape=(2*self._M+1,), dtype=object) for k in self.get_keys()}
        for k in self.get_keys():
            for m in np.arange(-self._M, self._M+1):
                new_fform_dict[k][n2i(m, self._M)] = self._fform_dict[k][n2i(m, self._M)] * (sp.cos(m * phase_shift) + 1j*sp.sin(m * phase_shift))
        self._fform_dict = new_fform_dict

    def get_M(self):
        return self._M

    def get_pc(self, pc):
        return np.copy(self._fform_dict[pc])
        
    def get_keys(self):
        return list(self._fform_dict.keys())

    def get_values(self):
        return list(self._fform_dict.values())

    def get_items(self):
        return list(self._fform_dict.items())


class FormContainer:

    def __init__(self, iterable):
        self._check_defining_iterable
        self._form_dict = dict(iterable)
        self._sympify

    def get_free_symbols(self):
        free_symbols = []
        for k in self.get_keys():
            for fs in self._form_dict[k].free_symbols:
                free_symbols.append(fs)
        free_symbols = list(set(free_symbols))
        free_symbols.sort(key=str)
        return free_symbols

    def subs(self, subs_array):
        subs_form_dict = {}
        for k,v in self.get_items():
            subs_form_dict[k] = self._form_dict[k].subs(subs_array)
        self._form_dict = subs_form_dict

    def _check_defining_iterable(self, iterable):
        def raise_error():
            raise ValueError('Invalid Data input.')
        try:
            bad_key = False in [type(k) == str for k in dict(iterable).keys()]
        except:
            raise_error()
        if bad_key:
            raise_error()

    def _sympify(self):
        for k in self.get_keys():
            self._form_dict[k] = sp.sympify(self._form_dict[k])

    def simplify(self):
        for k,v in self.get_items():
            self._form_dict[k] = sp.simplify(v)

    def get_keys(self):
        return list(self._form_dict.keys())

    def get_values(self):
        return list(self._form_dict.values())

    def get_items(self):
        return list(self._form_dict.items())


def n2i(n, M=16):
    return n+M


def fform_to_fdat(fform, subs_dict):
    subs_array = [(k,subs_dict[k]) for k in fform.get_free_symbols()]
    new_fform = deepcopy(fform)
    new_fform.subs(subs_array)
    iterable = {k:v.astype(complex) for k,v in new_fform.get_items()}
    return fDataContainer(iterable, new_fform.get_M())
    

def fdat_to_dat(fdat, num_points): 
    iterable = {}
    M = fdat.get_M()
    xdata = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    for k,v in fdat.get_items():
        ydata = np.zeros(num_points, dtype=complex)
        for m in np.arange(-M, M+1):
            ydata += v[n2i(m, M)] * (np.cos(m * xdata) + 1j*np.sin(m * xdata))
        iterable[k] = np.array([xdata, ydata]).real.astype(float)
    return DataContainer(iterable, 'radians')


def fform_to_dat(fform, subs_dict, num_points):
    return fdat_to_dat(fform_to_fdat(fform, subs_dict), num_points)


def fform_to_form(fform):
    iterable = {}
    for k,v in fform.get_items():
        iterable[k] = formula_from_fexpr(v, fform.get_M())
    return FormContainer(iterable)


def formula_from_fexpr(t, M=16):
    expr = 0
    for m in np.arange(-M, M+1):
        expr += t[n2i(m)]*(sp.cos(m*S.phi)+1j*sp.sin(m*S.phi))
    return expr


def form_to_dat(form, subs_array, num_points):
    new_form = deepcopy(form)
    new_form.subs(subs_array)
    if new_form.get_free_symbols() != [S.phi]:
        raise ValueError('Only one variable allowed in conversion from form to dat. Is subs_array correct?')
    iterable = {}
    for k,v in new_form.get_items():
        f = sp.lambdify(S.phi, v)
        xdata = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        ydata = f(xdata)
        iterable[k] = np.array([xdata, ydata], dtype=complex).real
    return DataContainer(iterable, 'radians')


def read_csv_file(filename, delimiter=','):
    xdata = []
    ydata = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            xdata.append(float(row[0]))
            ydata.append(float(row[1]))
    return np.array([xdata, ydata])


def dat_subtract(dat1, dat2):
    if set(dat1.get_keys()) != set(dat2.get_keys()):
        raise ValueError('DataContainers have different keys.')
    if dat1.get_values('radians')[0].shape != dat2.get_values('radians')[0].shape:
        raise ValueError('DataContainers\' xydatas have different shapes.')
    new_dict = {}
    for k in dat1.get_keys():
        new_xdata, ydata1 = dat1.get_pc(k, 'radians')
        _, ydata2 = dat2.get_pc(k, 'radians')
        if not np.array_equal(new_xdata, _):
            raise ValueError('DataContainers have different xdatas.')
        new_ydata = ydata2 - ydata1
        new_dict[k] = np.array([new_xdata, new_ydata])
    return DataContainer(new_dict, 'radians')
       

def load_data_and_dark_subtract(data_filenames_dict, data_angle_units, dark_filenames_dict, dark_angle_units):
    if set(data_filenames_dict.keys()) != set(dark_filenames_dict.keys()):
        raise ValueError('filename dicts have different keys.')
    dat1 = DataContainer({k:read_csv_file(v) for k,v in data_filenames_dict.items()}, data_angle_units)
    dat2 = DataContainer({k:read_csv_file(v) for k,v in dark_filenames_dict.items()}, dark_angle_units)
    return dat_subtract(dat1, dat2)
    

def load_data(data_filenames_dict, angle_units):
    return DataContainer({k:read_csv_file(v) for k,v in data_filenames_dict.items()}, angle_units)


def data_dft(dat, interp_kind='cubic', M=16):
    ans = {pc:np.zeros(2*M+1, dtype=np.complex64) for pc in dat.get_keys()}
    for k in dat.get_keys():
        for m in np.arange(-M, M+1):
            xdata, ydata = dat.get_pc(k, 'radians')
            interp_func = interp1d(xdata, ydata, kind=interp_kind)
            interp_xdata = np.linspace(0, 2*np.pi, len(ydata), endpoint=False)
            interp_ydata = interp_func(interp_xdata)
            dx = interp_xdata[1] - interp_xdata[0]
            ans[k][n2i(m, M)] = sum([1/2/np.pi*dx*interp_ydata[i]*np.exp(-1j*m*interp_xdata[i]) for i in range(len(interp_xdata))])
    return fDataContainer(ans.items(), M=M)


def load_data_and_fourier_transform(data_filenames_dict, data_angle_units, dark_filenames_dict=None, dark_angle_units=None, interp_kind='cubic', M=16, min_subtract=False, scale=1):
    if dark_filenames_dict is not None:
        dat = load_data_and_dark_subtract(data_filenames_dict, data_angle_units, dark_filenames_dict, dark_angle_units)
    else:
        dat = load_data(data_filenames_dict, data_angle_units)
    if min_subtract:
        dat.subtract_min()
    fdat = data_dft(dat, interp_kind=interp_kind, M=M)
    if scale != 1:
        dat.scale_data(scale)
        fdat.scale_data(scale)
    return dat, fdat


def load_fform(filename):
    with open(filename, 'rb') as f:
        fform_dict = pickle.load(f)
    return fFormContainer(fform_dict)
        

    











