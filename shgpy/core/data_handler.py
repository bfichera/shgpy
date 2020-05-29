import numpy as np
import csv

from .. import fourierutils as fx


class DataContainer:

    def __init__(self, iterable, input_angle_units):
        self._input_angle_units = input_angle_units.lower()
        self._check_angle_units(input_angle_units)
        self._check_defining_iterable(iterable)
        self._input_angle_units = input_angle_units
        self._scale = scale
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
        offset -= minval

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

    def get_xydata(self, pc, requested_angle_units):
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
                ans[fx.n2i(m, self._M)] = fexpr[fx.n2i(m, self._M)] * (np.cos(m * angle)+1j*np.sin(m * angle))
            self._fdata_dict[k] = ans
        self._phase_shift += angle


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
    if dat1.get_values()[0].shape != dat2.get_values()[0].shape:
        raise ValueError('DataContainers\' xydatas have different shapes.')
    new_dict = {}
    for k in dat1.get_keys():
        new_xdata, ydata1 = dat1.get_xydata(k, 'radians')
        _, ydata2 = dat2.get_xydata(k, 'radians')
        new_ydata = ydata2-ydata1
        new_dict[k] = np.array([new_xdata, new_ydata])
    return DataContainer(new_dict[k], 'radians')
       

def load_data_and_dark_subtract(data_filenames_dict, dark_filenames_dict, data_angle_units, dark_angle_units):
    if set(data_filenames_dict.get_keys()) != set(dark_filenames_dict.get_keys()):
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
            xdata, ydata = dat.get_xydata(k, 'radians')
            interp_func = interp1d(xdata, ydata, kind=interp_kind)
            interp_xdata = np.linspace(0, 2*np.pi, len(ydata), endpoint=False)
            interp_ydata = interp_func(interp_xdata)
            dx = interp_xdata[1] - interp_xdata[0]
            ans[k][n2i(m, M)] = sum([1/2/np.pi*dx*interp_ydata[i]*np.exp(-1j*m*interp_xdata[i]) for i in range(len(interp_xdata))])
    return shgpy.fDataContainer(ans.items(), M=M)


def load_data_and_fourier_transform(data_filenames_dict, data_angle_units, dark_filenames_dict=None, dark_angle_units=None, interp_kind='cubic', M=16, min_subtract=False, scale_factor=None)
    if dark_filenames_dict is not None:
        dat = load_data_and_dark_subtract(data_filenames_dict, dark_filenames_dict, data_angle_units, dark_angle_units)
    else:
        dat = load_data(data_filenames_dict, data_angle_units)
    if min_subtract:
        dat.subtract_min()
    fdat = data_dft(dat, interp_kind=interp_kind, M=M)
    if scale_factor is not None:
        dat.scale_data(scale)
        fdat.scale_data(scale)
    return dat, fdat


    










