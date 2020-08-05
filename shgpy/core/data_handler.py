import numpy as np
import sympy as sp
import csv
import pickle
from scipy.interpolate import interp1d
from scipy.integrate import quad
import logging
import itertools
from copy import deepcopy
from .. import shg_symbols as S
from warnings import warn

_logger = logging.getLogger(__name__)


class DataContainer:
    """Contains SHG data in `phi`-space.

    Parameters
    ----------
    iterable : dict or array_like
        dict which defines the data. Should be in form
        `{'PP':array([xdata, ydata]), 'PS':...}`. Also accepts `array_like`
        of form `(('PP':array([xdata, ydata])), ('PS', ...), ...)`.
    input_angle_units : {'radians', 'degrees'}
        Units of `xdata`.
    normal_to_oblique : bool, optional
        Defaults to ``False``. If ``True``, iterable should have only
        two elements: one with key 'PP' or 'SS', and one with key 'PS'
        or 'SP'. Then, upon initialization, compute the "missing"
        polarization components by rotating by 90 degrees. The final
        object has ``dat.get_keys() = ['PP', 'PS', 'SP', 'SS']``.

    Notes
    -----
    The function of the `DataContainer` class is to hold data which could
    just as well be contained in a dictionary defined like
    ``data = {'PP':array([xdata, ydata]), 'PS':array([xdata, ydata]), ...}``.
    Many of the associated methods involve getting and setting this data like
    you would in a dictionary. However, putting this data in a predefined objects
    allows one to keep track of whatever manipulations (scaling, offsetting) have
    been applied to the data, as well as other attributes like the units with
    which the data was defined.

    """
    def __init__(self, iterable, input_angle_units, normal_to_oblique=False):
        self._input_angle_units = input_angle_units.lower()
        self._check_angle_units(input_angle_units)
        self._check_defining_iterable(iterable)
        self._input_angle_units = input_angle_units
        self._scale = 1
        self._data_dict = deepcopy(dict(iterable))
        self._phase_shift = 0
        self._offset = 0
        if input_angle_units == 'degrees':
            self._convert_data_dict_to_radians()
        if normal_to_oblique:
            self._coerce_to_oblique_incidence()
        self._remove_duplicates()
        self._modulate_data_dict()
        self._sort_data_dict()

    def _check_angle_units(self, angle_units):
        if angle_units not in ['radians', 'degrees']:
            raise ValueError('angle_units input must be one of \'radians\' or \'degrees\'.')

    def _check_defining_iterable(self, iterable):
        try:
            bad_key = False in [type(k) == str for k in dict(iterable).keys()]
            bad_val = False in [type(v) == np.ndarray and v[0].shape == v[1].shape and v.dtype != object and v.ndim == 2 for v in dict(iterable).values()]
        except:
            raise ValueError('Invalid data input')
        if bad_key or bad_val:
            raise ValueError(f'Invalid data input: bad {"key" if bad_key else "value"}')

    def _remove_duplicates(self):
        for k,v in self._data_dict.items():
            new_xdata = []
            new_ydata = []
            for x,y in zip(v[0], v[1]):
                if not (x % (2*np.pi)) in new_xdata:
                    new_xdata.append(x)
                    new_ydata.append(y)
            if len(new_xdata) != len(v[0]):
                warn('Duplicate values encountered in data file.')
            new_xydata = list(zip(*zip(new_xdata, new_ydata)))
            self._data_dict[k] = np.array(new_xydata, dtype=v.dtype)

    def _coerce_to_oblique_incidence(self): 
        keys = set((k.upper()[:2] for k in self._data_dict.keys()))
        if keys in [set(['PP', 'PS', 'SP', 'SS'])]:
            key_type = 'oblique'
        elif keys in [set(i) for i in itertools.product(['PP', 'SS'], ['PS', 'SP'])]:
            key_type = 'normal'
        else:
            raise ValueError("Invalid input: keys must be 'PP', 'PS', 'SP', 'SS' or a nonparallel combination of two.")
        
        if key_type == 'oblique':
            self._data_dict = {k.upper()[:2]:self._data_dict[k] for k in self._data_dict.keys()}
        if key_type == 'normal':
            self._data_dict = {k.upper()[:2]:self._data_dict[k] for k in self._data_dict.keys()}
            leftover = ['PP', 'PS', 'SP', 'SS']
            for k in self._data_dict.keys():
                leftover.remove(k)
            for k in leftover:
                if k[0] == k[1]:
                    copy_pc = {'P':'S', 'S':'P'}[k[0]]*2
                else:
                    copy_pc = k[1]+k[0]
                self._data_dict[k] = deepcopy(self._data_dict[copy_pc])
                for i,x in enumerate(self._data_dict[k][0]):
                    self._data_dict[k][0][i] = (x + np.pi/2) % (2*np.pi)

    def _sort_data_dict(self):
        for k,v in self._data_dict.items():
            new_v = np.array(list(zip(*sorted(zip(v[0], v[1]), key=lambda x:x[0]))))
            self._data_dict[k] = new_v

    def _modulate_data_dict(self):
        for k,v in self._data_dict.items():
            xdata, ydata = v
            new_xdata = xdata % (2*np.pi)
            self._data_dict[k] = np.array([new_xdata, ydata], dtype=v.dtype)
                
    def _convert_data_dict_to_radians(self):
        self._data_dict = {k:np.array([np.deg2rad(v[0]), np.copy(v[1])]) for k,v in self._data_dict.items()}

    def _get_data_dict_degrees(self):
        return {k:np.array([np.rad2deg(v[0]), np.copy(v[1])]) for k,v in self._data_dict.items()}

    def scale_data(self, scale_factor):
        """Scale the data by a constant factor."""
        if scale_factor < 0:
            raise ValueError('scale factor cannot be negative.')
        self._data_dict = {k:np.array([v[0], scale_factor*v[1]]) for k,v in self._data_dict.items()}
        self._scale *= scale_factor

    def subtract_min(self):
        """Subtract the minimum from the data."""
        minval = self.get_minval()[1]
        for k,v in self.get_items('radians'):
            self._data_dict[k] = np.array([v[0], v[1]-minval])
        self._offset -= minval

    def get_maxval(self):
        """Get the maximum value of the data.
        
        Returns
        -------
        mac_pc : str
            The polarization combination for which the maximum occurred
        maximum_value : float
            The maximum value attained.
        """
        maximum_value = -np.inf
        max_pc = None
        for k,v in self._data_dict.items():
            if max(v[1]) > maximum_value:
                maximum_value = max(v[1])
                max_pc = k
        return max_pc, maximum_value

    def get_pc_maxval(self, pc):
        """Get the maximum value of a particular polarization combination.
        
        Parameters
        ----------
        pc : str
            The polarization combination of interest.
        
        Returns
        -------
        maximum_value : float
        """
        return max(self._data_dict[pc][1])

    def get_minval(self):
        """Get the minimum value of the data.

        Returns
        -------
        min_pc : str
            The polarization combination for which the minimum occurred.
        minimum_value : float
            The minimum value attained.
        """
        minimum_value = np.inf
        min_pc = None
        for k,v in self._data_dict.items():
            if min(v[1]) < minimum_value:
                minimum_value = min(v[1])
                min_pc = k
        return min_pc, minimum_value

    def get_pc_minval(self, pc):
        """Get the minimum value of a particular polarization combination.
        
        Parameters
        ----------
        pc : str
            The polarization combination of interest.
        
        Returns
        -------
        minimum_value : float
        """
        return min(self._data_dict[pc][1])

    def get_offset(self):
        """Returns the degree to which the data has been offset"""
        return self._offset

    def normalize_data(self, desired_maximum=1):
        """Normalize the data to a desired maximum."""
        _, maximum_value = self.get_maxval()
        self.scale_data(desired_maximum / maximum_value)

    def phase_shift_data(self, angle, angle_units):
        """Phase shift the data by a constant amount.
        
        Parameters
        ----------
        angle : float
        angle_units : {'radians', 'degrees'}
            Units of argument `angle`.
        """
        angle_units = angle_units.lower()
        self._check_angle_units(angle_units)
        if angle_units == 'degrees':
            angle = np.deg2rad(angle)
        self._data_dict = {k:np.array([(v[0]+angle) % (2*np.pi), v[1]]) for k,v in self._data_dict.items()}
        self._phase_shift += angle
        self._sort_data_dict()

    def get_pc(self, pc, requested_angle_units):
        """Get data for single polarization combination.

        Parameters
        ----------
        pc : str
            Polarization combination of interest.
        requested_angle_units : {'radians', 'degrees'}
            Units requested for `xdata`.

        Returns
        -------
        data : ndarray
            Data for polarization combination `pc` in the form
            `array([xdata, ydata])`.
        """
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return np.copy(self._data_dict[pc])
        if requested_angle_units == 'degrees':
            return np.copy(self._get_data_dict_degrees()[pc])

    def get_keys(self):
        """Get the polarization combinations for this data.

        Equivalent to `dict.keys()`.
        
        Returns
        -------
        pcs : list of str
        """
        return list(self._data_dict.keys())

    def get_values(self, requested_angle_units):
        """Get a list of data (without polarization combination reference)

        Equivalent to `dict.values()`.

        Parameters
        ----------
        requested_angle_units : {'radians', 'degrees'}
            Units requested for `xdata`.

        Returns
        -------
        data : list of ndarray
            Data for each polarization combination
            in the form `array([xdata, ydata])`
        """
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return list(self._data_dict.values())
        elif requested_angle_units == 'degrees':
            return list(self._get_data_dict_degrees().values())

    def get_items(self, requested_angle_units):
        """Get a list of `(pc, data)` pairs.

        Equivalent to `dict.items()`.

        Parameters
        ----------
        requested_angle_units : {'radians', 'degrees'}
            Units requested for `xdata`.

        Returns
        -------
        data : list of (pc, ndarray) tuple
            List of data for each polarization combination, each
            in the form `(pc, array([xdata, ydata]))`
        """
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return list(self._data_dict.items())
        elif requested_angle_units == 'degrees':
            return list(self._get_data_dict_degrees().items())

    def get_scale(self):
        """Get the extent to which this data has been scaled from the original."""
        return self._scale

    def get_phase_shift(self, requested_angle_units):
        """Get the extent to which this data has been phase shifted from the original.

        Parameters
        ----------
        requested_angle_units : {'radians', 'degrees'}

        Returns
        -------
        phase_shift : float
        """
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'radians':
            return self._phase_shift
        elif requested_angle_units == 'degrees':
            return np.rad2deg(self._phase_shift)


class fDataContainer:
    """Contains SHG data in Fourier space.

    Parameters
    ----------
    iterable : dict or array_like
        dict which defines the data. Should be in form
        `{'PP':ndarray(...), 'PS':..., ...}`. Also accepts `array_like`
        of form `(('PP':ndarray(...)), ('PS', ...), ...)`.

    Notes
    -----
    A central capability of the `shgpy` package is to be able to fit
    RA-SHG data in Fourier space. What that amounts to is fitting the
    Fourier components of a particular fitting formula to the Fourier
    components of the data. An instance of the `fDataContainer` class
    basically acts as a dictionary holding pairs `(pc, fdata)`, where
    `pc` is a `str` label for the polarization combination and `fdata`
    is an ndarray of length `2*M+1`. The `nth` element here denotes the
    `(n-M)th` fourier component of the data to be fitted. 
    """

    def __init__(self, iterable, M=16):
        self._M = M
        self._check_defining_iterable(iterable)
        self._fdata_dict = deepcopy(dict(iterable))
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
        """Get the polarization combinations for this data.

        Equivalent to `dict.keys()`.
        
        Returns
        -------
        pcs : list of str
        """
        return list(self._fdata_dict.keys())

    def get_values(self):
        """Get a list of data (without polarization combination reference)

        Equivalent to `dict.values()`.

        Returns
        -------
        fdata : list of ndarray
            Data for each polarization combination
            in the form `array(shape=(2*M+1,))`
        """
        return list(self._fdata_dict.values())

    def get_items(self):
        """Get a list of `(pc, fdata)` pairs.

        Equivalent to `dict.items()`.

        Returns
        -------
        fdata : list of (pc, ndarray) tuple
            List of data for each polarization combination, each
            in the form `(pc, array(shape=(2*M+1,)))`
        """
        return list(self._fdata_dict.items())

    def get_scale(self):
        """Get the extent to which this data has been scaled from the original."""
        return self._scale

    def get_M(self):
        """Get the number of Fourier frequencies.

        Each `fdata` array has `2*M+1` elements."""
        return self._M

    def get_phase_shift(self, requested_angle_units):
        """Get the extent to which this data has been phase shifted from the original.

        Parameters
        ----------
        requested_angle_units : {'radians', 'degrees'}

        Returns
        -------
        phase_shift : float
        """
        requested_angle_units = requested_angle_units.lower()
        self._check_angle_units(requested_angle_units)
        if requested_angle_units == 'degrees':
            return np.rad2deg(self._phase_shift)
        return self._phase_shift
        
    def scale_fdata(self, scale_factor):
        """Normalize the fdata by a given amount."""
        if scale_factor < 0:
            raise ValueError('scale factor cannot be negative.')
        self._fdata_dict = {k:scale_factor*v for k,v in self.get_items()}
        self._scale *= scale_factor

    def get_pc(self, pc):
        """Get fdata for single polarization combination.

        Parameters
        ----------
        pc : str
            Polarization combination of interest.

        Returns
        -------
        fdata : ndarray
            Data for polarization combination `pc` with shape `2*M+1`. 
        """
        return np.copy(self._fdata_dict[pc])

    def get_maxval(self):
        """Get the maximum value of the data.
        
        Returns
        -------
        mac_pc : str
            The polarization combination for which the maximum occurred
        maximum_value : float
            The maximum value attained.

        Notes
        -----
        Since the elements of the fourier transform are in general
        complex numbers, this method returns the maximum value of
        the magnitudes of the different elements.
        """
        maximum_value = -np.inf
        max_pc = None
        for k,v in self.get_items():
            this_max = max([abs(nv) for nv in v])
            if this_max > maximum_value:
                maximum_value = this_max
                max_pc = k
        return max_pc, maximum_value

    def get_pc_maxval(self, pc):
        """Get the maximum value of a particular polarization combination.
        
        Parameters
        ----------
        pc : str
            The polarization combination of interest.
        
        Returns
        -------
        maximum_value : float
            Returns maximum of magnitudes of fourier components
            (see :func:`~shgpy.core.data_handler.fDataContainer.get_maxval`)
        """
        v = self._fdata_dict[pc]
        maximum_value = max([abs(nv) for nv in v])
        return maximum_value
        
    def normalize_fdata(self, desired_maximum):
        """Normalize the data to a desired maximum."""
        _, maximum_value = self.get_maxval()
        self.scale_fdata(desired_maximum / maximum_value)

    def phase_shift_fdata(self, angle, angle_units):
        """Phase shift the data by a constant amount.
        
        Parameters
        ----------
        angle : float
        angle_units : {'radians', 'degrees'}
            Units of argument `angle`.
        """
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
    """Contains an SHG formula in Fourier space.

    Parameters
    ----------
    iterable : dict or array_like
        dict which defines the Fourier formula. Should be in form
        `{'PP':ndarray(...), 'PS':..., ...}`. Also accepts `array_like`
        of form `(('PP':ndarray(...)), ('PS', ...), ...)`.

    Notes
    -----
    In order to fit RA-SHG data in Fourier space, we need to know
    the different Fourier components of our fitting formula. An
    instance of the `fFormContainer` class provides a convenient
    means of holding and manipulating such a Fourier formula. It
    basically acts as a dictionary of `{pc:fformula}` pairs, where
    `pc` labels the polarization combination and `fformula` is an
    `ndarray` of length `2*M+1`. Here, the `nth` element of `fformula`
    contains a formula for the `(n-M)th` Fourier coefficient.
    """
    def __init__(self, iterable, M=16):
        self._M = M
        self._check_defining_iterable(iterable)
        self._fform_dict = self._copy_iterable(iterable)
        self._sympify()

    def _copy_iterable(self, iterable):
        _fform_dict = {
            k:np.array([s for s in v])
            for k,v in dict(iterable).items()
        }
        return _fform_dict

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
        """Return a sorted list of the free symbols in the Fourier formula."""
        free_symbols = []
        for k in self.get_keys():
            for m in np.arange(-self._M, self._M+1):
                for fs in self._fform_dict[k][n2i(m, self._M)].free_symbols:
                    free_symbols.append(fs)
        free_symbols = list(set(free_symbols))
        free_symbols.sort(key=str)
        return free_symbols

    def subs(self, subs_array):
        """Substitute for some variables in the Fourier formula.

        Parameters
        ----------
        subs_array : array_like of array_like of sympy.Expr
            An array of pairs of values. For example, to substitute the expression
            `2*xyz` for the variable `xxx`, use ``subs_array = ((xxx, 2*xyz))``.
        """
        subs_fform_dict = {}
        for k,v in self.get_items():
            subs_fform_dict[k] = np.zeros(shape=(2*self._M+1,), dtype=object)
            for m in np.arange(-self._M, self._M+1):
                subs_fform_dict[k][n2i(m, self._M)] = self._fform_dict[k][n2i(m, self._M)].subs(subs_array)
        self._fform_dict = subs_fform_dict

    def apply_phase_shift(self, phase_shift):
        """Phase shift the Fourier formula.
        
        Parameters
        ----------
        phase_shift : sympy.Expr 

        Notes
        -----
        This method works simply by multiplying the `mth` component of the
        Fourier formula by ``sympy.exp(1j * m * phase_shift)``.

        """
        new_fform_dict = {k:np.zeros(shape=(2*self._M+1,), dtype=object) for k in self.get_keys()}
        for k in self.get_keys():
            for m in np.arange(-self._M, self._M+1):
                new_fform_dict[k][n2i(m, self._M)] = self._fform_dict[k][n2i(m, self._M)] * (sp.cos(m * phase_shift) + 1j*sp.sin(m * phase_shift))
        self._fform_dict = new_fform_dict

    def get_M(self):
        """Get the number of Fourier frequencies.

        Each Fourier formula array has `2*M+1` elements."""
        return self._M

    def get_pc(self, pc):
        """Get the Fourier formula for single polarization combination.

        Parameters
        ----------
        pc : str
            Polarization combination of interest.

        Returns
        -------
        fformula : ndarray
            Fourier formula array for polarization combination `pc`. 
        """
        return np.copy(self._fform_dict[pc])
        
    def get_keys(self):
        """Get the polarization combinations for this Fourier formula.

        Equivalent to `dict.keys()`.
        
        Returns
        -------
        pcs : list of str
        """
        return list(self._fform_dict.keys())

    def get_values(self):
        """Get a list of the Fourier formulas (without polarization combination reference)
    
        Equivalent to `dict.values()`.

        Returns
        -------
        values : list of ndarray
            Data for each polarization combination
            in the form `array([xdata, ydata])`
        """
        return list(self._fform_dict.values())

    def get_items(self):
        """Get a list of `(pc, fformula)` pairs.

        Equivalent to `dict.items()`.

        Returns
        -------
        items : list of (str, ndarray) tuple
            List of data for each polarization combination, each
            in the form `(pc, array(shape=(2*M+1,)))`
        """
        return list(self._fform_dict.items())


class FormContainer:
    """Contains an SHG formula in `phi`-space.

    Parameters
    ----------
    iterable : dict or array_like
        dict which defines the formula. Should be in form
        `{'PP':sympy.Expr, 'PS':..., ...}`. Also accepts `array_like`
        of form `(('PP':sympy.Expr), ('PS', ...), ...)`.

    Notes
    -----
    While fitting RA-SHG in Fourier space is usually faster, for simple
    problems it is acceptable to fit in real space. An instance of the
    `FormContainer` class provides a convenient means of storing and
    manipulating such a formula. It basically acts as a dictionary of
    `{pc:formula}` pairs, where `pc` labels the polarization combination
    and `formula` is an instance of the sympy.Expr class.
    """

    def __init__(self, iterable):
        self._check_defining_iterable(iterable)
        self._form_dict = self._copy_iterable(iterable)
        self._sympify()

    def _copy_iterable(self, iterable):
        _form_dict = {
            k:v for k,v in dict(iterable).items()
        }
        return _form_dict

    def get_free_symbols(self):
        """Return a sorted list of the free symbols in the Fourier formula."""
        free_symbols = []
        for k in self.get_keys():
            for fs in self._form_dict[k].free_symbols:
                free_symbols.append(fs)
        free_symbols = list(set(free_symbols))
        free_symbols.sort(key=str)
        return free_symbols

    def subs(self, subs_array):
        """Substitute for some variables in the Fourier formula.

        Parameters
        ----------
        subs_array : array_like of array_like of sympy.Expr
            An array of pairs of values. For example, to substitute the expression
            `2*psi` for the variable `phi`, use ``subs_array = ((phi, 2*psi))``.
        """
        subs_form_dict = {}
        for k,v in self.get_items():
            subs_form_dict[k] = self._form_dict[k].subs(subs_array)
        self._form_dict = subs_form_dict
    
    def apply_phase_shift(self, phase_shift, var_to_shift):
        """Phase shift the formula.
        
        Parameters
        ----------
        phase_shift : sympy.Expr 
            Amount to shift by
        var_to_shift : sympy.Symbol
            Variable to shift

        Notes
        -----
        This method is equivalent to ``subs([var_to_shift, var_to_shift+phase_shift])``
        """
        self.subs([(var_to_shift, var_to_shift + phase_shift)])

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
        """Apply the `sympy.simplify` method to every polarization combination."""
        for k,v in self.get_items():
            self._form_dict[k] = sp.simplify(v)

    def expand(self):
        """Apply the `sympy.expand` method to every polarization combination."""
        for k,v in self.get_items():
            self._form_dict[k] = sp.expand(v)

    def get_keys(self):
        """Get the polarization combinations for this formula.

        Equivalent to `dict.keys()`.
        
        Returns
        -------
        pcs : list of str
        """
        return list(self._form_dict.keys())

    def get_values(self):
        """Get a list of the formulas (without polarization combination reference)
    
        Equivalent to `dict.values()`.

        Returns
        -------
        values : list of sympy.Expr
        """
        return list(self._form_dict.values())

    def get_items(self):
        """Get a list of `(pc, formula)` pairs.

        Equivalent to `dict.items()`.

        Returns
        -------
        items : list of (str, sympy.Expr) tuple
            List of data for each polarization combination, each
            in the form `(pc, formula)`
        """
        return list(self._form_dict.items())

    def get_pc(self, pc):
        """Get the formula for single polarization combination.

        Parameters
        ----------
        pc : str
            Polarization combination of interest.

        Returns
        -------
        formula : sympy.Expr
            Formula expression for polarization combination `pc`. 
        """
        return self._form_dict[pc]


def n2i(n, M=16):
    """Convert between Fourier index and array index

    Returns the index (0-`2*M+1`) in a Fourier array corresponding to the `nth` Fourier component
    """
    return n+M


def fform_to_fdat(fform, subs_dict):
    """Convert instance of `fFormContainer` to instance of `fDataContainer`

    Parameters
    ----------
    fform : fFormContainer
        Instance of :class:`~shgpy.core.data_handler.fFormContainer`
    subs_dict : dict
        Dict of `{sympy.Symbol : float}` pairs to perform the substitution.

    Returns
    -------
    fdat : fDataContainer
        Instance of `fDataContainer`

    Notes
    -----
    This function operates by taking an instance of the `fFormContainer`
    class and substituting for all the relevant `sympy.Symbol`s according
    to `subs_dict`.

    """
    subs_array = [(k,subs_dict[k]) for k in fform.get_free_symbols()]
    new_fform = fFormContainer(fform.get_items())
    new_fform.subs(subs_array)
    iterable = {k:v.astype(complex) for k,v in new_fform.get_items()}
    return fDataContainer(iterable, new_fform.get_M())
    

def fdat_to_dat(fdat, num_points): 
    """Convert instance of `fDataContainer` to instance of `DataContainer`

    Parameters
    ----------
    fdat : fDataContainer
        Instance of :class:`~shgpy.core.data_handler.fDataContainer`
    num_points : int
        Number of different points to put into the `DataContainer`.

    Returns
    -------
    dat : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`

    Notes
    -----
    This function operates by taking an instance of the `fDataContainer`
    class and performing an inverse Fourier transform.

    """
    iterable = {}
    M = fdat.get_M()
    xdata = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    for k,v in fdat.get_items():
        ydata = np.zeros(num_points, dtype=complex)
        for m in np.arange(-M, M+1):
            ydata += v[n2i(m, M)] * (np.cos(m * xdata) + 1j*np.sin(m * xdata))
        iterable[k] = np.array([xdata, ydata]).real.astype(float)
    return DataContainer(iterable, 'radians', normal_to_oblique=False)


def fform_to_dat(fform, subs_dict, num_points):
    """Convert instance of `fFormContainer` to instance of `DataContainer`

    Parameters
    ----------
    fform : fFormContainer
        Instance of :class:`~shgpy.core.data_handler.fFormContainer`
    subs_dict : dict
        Dict of `{sympy.Symbol : float}` pairs to perform the substitution.
    num_points : int
        Number of different points to put into the `DataContainer`.

    Returns
    -------
    dat : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`

    Notes
    -----
    This function operates by taking an instance of the `fFormContainer`
    class and substituting for all the relevant `sympy.Symbol`s according
    to `subs_dict`. Then it creates a `DataContainer` instance with 
    `num_points` points.

    """
    return fdat_to_dat(fform_to_fdat(fform, subs_dict), num_points)


def fform_to_form(fform):
    """Convert instance of `fFormContainer` to instance of `FormContainer`

    Parameters
    ----------
    fform : fFormContainer
        Instance of :class:`~shgpy.core.data_handler.fFormContainer`

    Returns
    -------
    form : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`

    Notes
    -----
    This function operates by taking an instance of the `fFormContainer`
    class and performing a Fourier transform.

    """
    iterable = {}
    for k,v in fform.get_items():
        iterable[k] = _formula_from_fexpr(v, fform.get_M())
    return FormContainer(iterable)


def _formula_from_fexpr(t, M=16):
    expr = 0
    for m in np.arange(-M, M+1):
        expr += t[n2i(m)]*(sp.cos(m*S.phi)+1j*sp.sin(m*S.phi))
    return expr


def _fexpr_n(expr_arr, n, precision=7):
    Rshape = expr_arr.shape
    expr_arrf = expr_arr.flatten()
    h = np.zeros(len(expr_arrf), dtype=object)
    for i in range(len(expr_arrf)):
        f_re = sp.lambdify(S.phi, 1/2/sp.pi*expr_arrf[i]*sp.cos(-1*(n*S.phi)))
        f_im = sp.lambdify(S.phi, 1/2/sp.pi*expr_arrf[i]*sp.sin(-1*(n*S.phi)))
        t_re,_ = quad(f_re, 0, 2*np.pi)
        t_im,_ = quad(f_im, 0, 2*np.pi)
        h[i] = round(t_re, precision)+round(t_im, precision)*1j
    h = h.reshape(Rshape)
    return h


def form_to_dat(form, subs_dict, num_points):
    """Convert instance of `FormContainer` to instance of `DataContainer`

    Parameters
    ----------
    form : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`
    subs_dict : dict
        Dict of `{sympy.Symbol : float}` pairs to perform the substitution.
    num_points : int
        Number of different points to put into the `DataContainer`.

    Returns
    -------
    dat : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`

    Notes
    -----
    This function operates by taking an instance of the `FormContainer`
    class and substituting for all the relevant `sympy.Symbol`s according
    to `subs_dict`. Then it creates a `DataContainer` instance with 
    `num_points` points.

    """
    subs_array = [(k,subs_dict[k]) for k in subs_dict.keys()]
    new_form = FormContainer(form.get_items())
    new_form.subs(subs_array)
    if new_form.get_free_symbols() != [S.phi]:
        raise ValueError('Only one variable allowed in conversion from form to dat. Is subs_array correct?')
    iterable = {}
    for k,v in new_form.get_items():
        f = sp.lambdify(S.phi, v)
        xdata = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        ydata = np.array([f(x) for x in xdata])
        iterable[k] = np.array([xdata, ydata], dtype=complex).real
    return DataContainer(iterable, 'radians', normal_to_oblique=False)


def _I_component(expr):
    return (expr-expr.subs(sp.I, 0)).subs(sp.I, 1)


def _no_I_component(expr):
    return expr.subs(sp.I, 0)


def form_to_fform(form, M=16):
    """Convert instance of `FormContainer` to instance of `fFormContainer`

    Parameters
    ----------
    form : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`
    M : int, optional
        Number of Fourier frequencies, defaults to `16`.

    Returns
    -------
    fform : fFormContainer
        Instance of :class:`~shgpy.core.data_handler.fFormContainer`

    Notes
    -----
    This function operates by taking an instance of the `FormContainer`
    class and performing a Fourier transform.

    """
    iterable = {}
    for k,v in form.get_items():
        iterable[k] = np.zeros((2*M+1,), dtype=object)
        _logger.info(f'Currently computing {k}.')
        for m in np.arange(-M, M+1):
            _logger.debug(f'Currently computing m={m}.')
            expr = sp.expand_trig(v*(sp.cos(-m*S.phi)+1j*sp.sin(-m*S.phi))).expand()
            expr_re = _no_I_component(expr)
            expr_im = _I_component(expr)
            iterable[k][n2i(m, M)] = 1/2/sp.pi*sp.integrate(expr_re, (S.phi, 0, 2*sp.pi)) + 1/2/sp.pi*sp.I*sp.integrate(expr_im, (S.phi, 0, 2*sp.pi))
    return fFormContainer(iterable, M=M)


def form_to_fdat(form, subs_dict, M=16):
    """Convert instance of `FormContainer` to instance of `fDataContainer`

    Parameters
    ----------
    fform : FormContainer
        Instance of :class:`~shgpy.core.data_handler.FormContainer`
    subs_dict : dict
        Dict of `{sympy.Symbol : float}` pairs to perform the substitution.

    Returns
    -------
    fdat : fDataContainer
        Instance of `fDataContainer`

    Notes
    -----
    This function operates by taking an instance of the `FormContainer`,
    doing a Fourier transform to a `fFormContainer` instance, and then
    substituting for all the relevant `sympy.Symbol`s according to
    `subs_dict`.

    """
    return fform_to_fdat(form_to_fform(form, M), subs_dict)


def merge_containers(containers, mapping):
    """Merge two contains with new keys according to `mapping`.

    Parameters
    ----------
    containers : list of Container_like
        List of Containers (DataContainer, fDataContainer,
        FormContainer, or fFormContainer). Must be the same type.
    mapping : function(key, index)
        Function which maps `key` of Container and `index`
        of a Container in `containers` to `str`.

    Returns
    -------
    container : Container_like
        Container object of type ``type(containers[0])`` with items
        ((k1, v1), (k2, v2), ...), where ki are the mapped keys of
        the containers and vi are the corresponding values.

    Examples
    --------
    >>> xdata = np.linspace(0, 2*np.pi, 1000)
    >>> ydata = np.sin(xdata)**2
    >>> iterable = {k:np.array([xdata, ydata]) for k in ['PP', 'PS']}
    >>> dat1 = DataContainer(iterable)
    >>> dat2 = DataContainer(iterable)
    >>> def mapping(key, index):
    >>>     return key+str(index)
    >>> dat3 = merge_containers([dat1, dat2], mapping)
    >>> dat3.get_keys():
    ['PP0', 'PS0', 'PP1', 'PS2']
    >>> dat3.get_values():
    [array([xdata, ydata]), array([xdata, ydata]), ...]

    """
    iterable = {}
    new_type = type(containers[0])
    if new_type == DataContainer:
        type_d = True
    else:
        type_d = False
    if new_type not in [DataContainer, fDataContainer, FormContainer, fFormContainer]:
        raise TypeError('Containers must be dype DataContainer, fDataContainer, FormContainer, or fFormContainer')
    for i,container in enumerate(containers):
        if type_d:
            items = container.get_items('radians')
        else:
            items = container.get_items()
        for k,v in items:
            new_k = mapping(k, i)
            new_v = np.copy(v)
            iterable[new_k] = new_v
            if type(container) != new_type:
                raise TypeError('Containers are of different types.')
    
    if type_d:
        return new_type(iterable, 'radians')
    else:
        return new_type(iterable)


def read_csv_file(filename, delimiter=','):
    """Read a csv file.

    Parameters
    ----------
    filename : str or file object
    delimiter : str
        Defaults to ``','``.

    Returns
    -------
    data : ndarray
        Data from csv file of the form `array([xdata, ydata])`.

    """
    xdata = []
    ydata = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            xdata.append(float(row[0]))
            ydata.append(float(row[1]))
    return np.array([xdata, ydata])


def dat_subtract(dat1, dat2):
    """Subtract two `DataContainer` instances.

    Parameters
    ----------
    dat1 : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`
    dat2 : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`

    Returns
    -------
    dat_result : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer` containing
        the contents of `dat1-dat2`.
    
    """
    if set(dat1.get_keys()) != set(dat2.get_keys()):
        raise ValueError('DataContainers have different keys.')
    if dat1.get_values('radians')[0].shape != dat2.get_values('radians')[0].shape:
        raise ValueError('DataContainers\' xydatas have different shapes.')
    new_dict = {}
    for k in dat1.get_keys():
        new_xdata, ydata1 = dat1.get_pc(k, 'radians')
        _, ydata2 = dat2.get_pc(k, 'radians')
        if False in np.isclose(new_xdata.flatten(), _.flatten()):
            raise ValueError('DataContainers have different xdatas.')
        new_ydata = ydata1 - ydata2
        new_dict[k] = np.array([new_xdata, new_ydata])
    return DataContainer(new_dict, 'radians', normal_to_oblique=False)


def dat_add(dat1, dat2):
    """Add two `DataContainer` instances.

    Parameters
    ----------
    dat1 : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`
    dat2 : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`

    Returns
    -------
    dat_result : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer` containing
        the contents of `dat1+dat2`.
    
    """
    if set(dat1.get_keys()) != set(dat2.get_keys()):
        raise ValueError('DataContainers have different keys.')
    if dat1.get_values('radians')[0].shape != dat2.get_values('radians')[0].shape:
        raise ValueError('DataContainers\' xydatas have different shapes.')
    new_dict = {}
    for k in dat1.get_keys():
        new_xdata, ydata1 = dat1.get_pc(k, 'radians')
        _, ydata2 = dat2.get_pc(k, 'radians')
        if False in np.isclose(new_xdata.flatten(), _.flatten()):
            raise ValueError('DataContainers have different xdatas.')
        new_ydata = ydata1 + ydata2
        new_dict[k] = np.array([new_xdata, new_ydata])
    return DataContainer(new_dict, 'radians', normal_to_oblique=False)
       

def load_data_and_dark_subtract(data_filenames_dict, data_angle_units, dark_filenames_dict, dark_angle_units, normal_to_oblique=False):
    """Load RA-SHG data from a dict of filenames and dark subtract.

    Parameters
    ----------
    data_filenames_dict : dict of (str : str or file object)
        dict of form (polarization combination : filename).
    data_angle_units : {'radians', 'degrees'}
        Units of `xdata`
    dark_filenames_dict : dict of (str : str or file object)
        dict of form (polarization combination : filename).
    dark_angle_units : {'radians', 'degrees'}
        Units of `xdata` for the dark.
    normal_to_oblique : bool, optional
        Defaults to False. See :class:`~shgpy.core.data_handler.DataContainer`.

    Returns
    -------
    dat : DataContainer
    
    """
    if set(data_filenames_dict.keys()) != set(dark_filenames_dict.keys()):
        raise ValueError('filename dicts have different keys.')
    dat1 = DataContainer({k:read_csv_file(v) for k,v in data_filenames_dict.items()}, data_angle_units, normal_to_oblique=normal_to_oblique)
    dat2 = DataContainer({k:read_csv_file(v) for k,v in dark_filenames_dict.items()}, dark_angle_units, normal_to_oblique=normal_to_oblique)
    return dat_subtract(dat1, dat2)
    

def load_data(data_filenames_dict, angle_units, normal_to_oblique=False):
    """Load RA-SHG data from a dict of filenames.

    Parameters
    ----------
    data_filenames_dict : dict of (str : str or file object)
        dict of form (polarization combination : filename).
    data_angle_units : {'radians', 'degrees'}
        Units of `xdata`
    normal_to_oblique : bool, optional
        Defaults to False. See :class:`~shgpy.core.data_handler.DataContainer`.

    Returns
    -------
    dat : DataContainer
    
    """
    return DataContainer({k:read_csv_file(v) for k,v in data_filenames_dict.items()}, angle_units, normal_to_oblique=normal_to_oblique)


def dat_to_fdat(dat, interp_kind='cubic', M=16):
    """Convert instance of `DataContainer` to instance of `fDataContainer`

    Parameters
    ----------
    dat : DataContainer
        Instance of :class:`~shgpy.core.data_handler.DataContainer`
    interp_kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’,
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
        ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer
        to a spline interpolation of zeroth, first, second or third order;
        ‘previous’ and ‘next’ simply return the previous or next value of
        the point) or as an integer specifying the order of the spline
        interpolator to use. Default is ‘cubic’.
    M : int, optional
        Number of Fourier frequencies, defaults to `16`.

    Returns
    -------
    fdat : fDataContainer
        Instance of `fDataContainer`

    Notes
    -----
    This function works by taking a DataContainer instance and performing
    an interpolation so that the points are equally spaced along `phi`.
    Then it performs a Fourier transform and returns an fDataContainer
    instance. The interpolation is necessary because a large portion
    of the codebase requires equally spaced points in order for the Fourier
    transform to be well-defined.

    """
    ans = {pc:np.zeros(2*M+1, dtype=np.complex64) for pc in dat.get_keys()}
    for k in dat.get_keys():
        for m in np.arange(-M, M+1):
            xdata, ydata = dat.get_pc(k, 'radians')
            interp_func = interp1d(xdata, ydata, kind=interp_kind, fill_value='extrapolate')
            interp_xdata = np.linspace(0, 2*np.pi, len(ydata), endpoint=False)
            interp_ydata = interp_func(interp_xdata)
            dx = interp_xdata[1] - interp_xdata[0]
            ans[k][n2i(m, M)] = sum([1/2/np.pi*dx*interp_ydata[i]*np.exp(-1j*m*interp_xdata[i]) for i in range(len(interp_xdata))])
    return fDataContainer(ans.items(), M=M)


def load_data_and_fourier_transform(data_filenames_dict, data_angle_units, dark_filenames_dict=None, dark_angle_units=None, interp_kind='cubic', M=16, min_subtract=False, scale=1, normal_to_oblique=False):
    """Load RA-SHG data from a dict of filenames and Fourier transform

    Parameters
    ----------
    data_filenames_dict : dict of (str : str or file object)
        dict of form (polarization combination : filename).
    data_angle_units : {'radians', 'degrees'}
        Units of `xdata`
    dark_filenames_dict : dict of (str : str or file object)
        dict of form (polarization combination : filename).
    dark_angle_units : {'radians', 'degrees'}
        Units of `xdata` for the dark.
    interp_kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’,
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
        ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer
        to a spline interpolation of zeroth, first, second or third order;
        ‘previous’ and ‘next’ simply return the previous or next value of
        the point) or as an integer specifying the order of the spline
        interpolator to use. Default is ‘cubic’
        Dict of `{sympy.Symbol : float}` pairs to perform the substitution.
    M : int, optional
        Number of Fourier frequencies, defaults to `16`.
    min_subtract : bool
        Whether to subtract the minimum before Fourier transforming.
        Default is False.
    normal_to_oblique : bool, optional
        Defaults to False. See :class:`~shgpy.core.data_handler.DataContainer`.

    Returns
    -------
    dat : DataContainer
    fdat : fDataContainer
    
    """
    if dark_filenames_dict is not None:
        dat = load_data_and_dark_subtract(data_filenames_dict, data_angle_units, dark_filenames_dict, dark_angle_units, normal_to_oblique=normal_to_oblique)
    else:
        dat = load_data(data_filenames_dict, data_angle_units, normal_to_oblique=normal_to_oblique)
    if min_subtract:
        dat.subtract_min()
    fdat = dat_to_fdat(dat, interp_kind=interp_kind, M=M)
    if scale != 1:
        dat.scale_data(scale)
        fdat.scale_data(scale)
    return dat, fdat


def load_fform(fform_filename):
    """Creat instance of fFormContainer from a fform_filename.

    ``fform_filename`` s are generated using the utilities provided
    in the :mod:`~shgpy.fformgen` module. Those utilities output
    a pickled fFormContainer-like object which can be loaded into
    a true fFormContainer instance using this function.

    Parameters
    ----------
    fform_filename : str or file object

    """
    with open(fform_filename, 'rb') as fh:
        str_fform_dict = pickle.load(fh)
    _fform_dict = {
        k:np.array([sp.sympify(s) for s in v])
        for k,v in str_fform_dict.items()
    }
    return fFormContainer(_fform_dict)
