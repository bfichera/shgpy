import unittest

import shgpy
import numpy as np


class TestMergeContainers(unittest.TestCase):

    data_filenames_dict = {
        'PP':'tests/Data/dataPP.csv',
        'PS':'tests/Data/dataPS.csv',
        'SP':'tests/Data/dataSP.csv',
        'SS':'tests/Data/dataSS.csv',
    }
    fform_filename = 'tests/fform/T_d-S_2-S_2(110).p'

    def test_dat(self):
        dat1 = shgpy.load_data(self.data_filenames_dict, 'degrees')
        dat2 = shgpy.load_data(self.data_filenames_dict, 'degrees')

        def mapping(key, index):
            return str(index) + key

        dat3 = shgpy.merge_containers([dat1, dat2], mapping)
        iterable = {}
        for k,v in dat1.get_items('radians'):
            iterable['0'+k] = np.copy(v)
        for k,v in dat2.get_items('radians'):
            iterable['1'+k] = np.copy(v)
        dat3_compare = shgpy.DataContainer(iterable, 'radians')

        for k in dat3.get_keys():
            a = dat3.get_pc(k, 'radians')
            b = dat3_compare.get_pc(k, 'radians')
            for i in range(2):
                self.assertEqual(str(a[i]), str(b[i]))

    def test_fform(self):
        fform1 = shgpy.load_fform(self.fform_filename)
        fform2 = shgpy.load_fform(self.fform_filename)

        def mapping(key, index):
            return str(index) + key

        fform3 = shgpy.merge_containers([fform1, fform2], mapping)
        iterable = {}
        for k,v in fform1.get_items():
            iterable['0'+k] = np.copy(v)
        for k,v in fform2.get_items():
            iterable['1'+k] = np.copy(v)
        fform3_compare = shgpy.fFormContainer(iterable)

        for k in fform3.get_keys():
            a = fform3.get_pc(k)
            b = fform3_compare.get_pc(k)
            self.assertEqual(str(a), str(b))

