import numpy as np

from .. import fourierutils as fx


class Data:

    def __init__(self, iterable, input_angle_units):
        self._input_angle_units = input_angle_units.lower()
        self._check_angle_units(input_angle_units)
        self._check_defining_iterable(iterable)
        self._input_angle_units = input_angle_units
        self._scale = 1
        self._data_dict = dict(iterable)
        self._phase_shift = 0
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

    def get_maxval(self):
        maximum_value = -np.inf
        max_pc = None
        for k,v in self._data_dict.items():
            if max(v[1]) > maximum_value:
                maximum_value = max(v[1])
                max_pc = k
        return max_pc, maximum_value

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


class fData:

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


class fForm:

    def __init__(self, iterable):
        self._fform_dict = dict(iterable)












