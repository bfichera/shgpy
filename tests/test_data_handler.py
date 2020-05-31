import unittest

import shgpy
import numpy as np


class TestData(unittest.TestCase):

    def test_creation(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.DataContainer(items, 'radians')
        self.assertIsInstance(hi, shgpy.DataContainer)

    def test_dict_funcs(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.DataContainer(items, 'radians')
        self.assertEqual(hi.get_keys(), [i[0] for i in items])
        self.assertEqual(hi.get_items('radians'), list(items))
        self.assertEqual(hi.get_values('radians'), [i[1] for i in items])

    def test_manipulation(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.DataContainer(items, 'radians')
        hi.scale_data(100)
        self.assertEqual(hi.get_scale(), 100)
        hi.phase_shift_data(10, angle_units='degreeS')
        self.assertEqual(hi.get_phase_shift('radians'), np.deg2rad(10))

    def test_maxval(self):
        items = (('PP', np.zeros(shape=(3,3))), ('PS', np.ones(shape=(4,4))))
        hi = shgpy.DataContainer(items, 'degrees')
        pc, maxval = hi.get_maxval()
        self.assertEqual((pc, maxval), ('PS', 1))


class TestfData(unittest.TestCase):

    def test_creation(self):
        M = 16
        items = (('PP', np.zeros(2*M+1, dtype=complex)), ('PS', np.ones(2*M+1, dtype=complex)*4))
        hi = shgpy.fDataContainer(items, M=M)
        self.assertIsInstance(hi, shgpy.fDataContainer)

    def test_dict_funcs(self):
        M = 16
        items = (('PP', np.zeros(2*M+1, dtype=complex)), ('PS', np.ones(2*M+1, dtype=complex)*4))
        hi = shgpy.fDataContainer(items, M=M)
        self.assertEqual(hi.get_keys(), list(dict(items).keys()))
        self.assertEqual(hi.get_items(), list(dict(items).items()))
        self.assertEqual(hi.get_values(), list(dict(items).values()))

    def test_manipulation(self):
        M = 16
        items = (('PP', np.zeros(2*M+1, dtype=complex)), ('PS', np.ones(2*M+1, dtype=complex)*4))
        hi = shgpy.fDataContainer(items, M=M)
        hi.scale_fdata(14)
        self.assertEqual(hi.get_scale(), 14)
        hi.normalize_fdata(1)
        self.assertEqual(hi.get_scale(), 0.25)
        self.assertEqual(hi.get_maxval(), ('PS', 1.0))
        hi.phase_shift_fdata(14, 'degrees')
        self.assertEqual(hi.get_phase_shift('radians'), np.deg2rad(14))


class TestLoadData(unittest.TestCase):

    filenames_dict = {
        'PP':'tests/Data/dataPP.csv',
        'PS':'tests/Data/dataPS.csv',
        'SP':'tests/Data/dataSP.csv',
        'SS':'tests/Data/dataSS.csv',
    }
    
    def test_loaders(self):
        dat = shgpy.load_data(self.filenames_dict, 'degrees')
        self.assertIsInstance(dat, shgpy.DataContainer)
        dat = shgpy.load_data_and_dark_subtract(self.filenames_dict, 'degrees', self.filenames_dict, 'degrees')
        self.assertIsInstance(dat, shgpy.DataContainer)
        self.assertEqual(dat.get_maxval()[1], 0)
        dat, fdat = shgpy.load_data_and_fourier_transform(self.filenames_dict, 'degrees')
        self.assertNotEqual(fdat.get_maxval()[1], 0)
        dat, fdat = shgpy.load_data_and_fourier_transform(self.filenames_dict, 'degrees', self.filenames_dict, 'degrees')
        self.assertEqual(fdat.get_maxval()[1], 0)
    
        
        







