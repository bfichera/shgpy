import unittest

import shgpy
import numpy as np


class TestData(unittest.TestCase):

    def test_creation(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.Data(items)
        self.assertIsInstance(hi, shgpy.Data)

    def test_dict_funcs(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.Data(items)
        self.assertEqual(hi.get_keys(), [i[0] for i in items])
        self.assertEqual(hi.get_items(), list(items))
        self.assertEqual(hi.get_values(), [i[1] for i in items])

    def test_manipulation(self):
        items = (('PS', np.ones((2,2))),)
        hi = shgpy.Data(items)
        hi.scale_data(100)
        self.assertEqual(hi.get_scale(), 100)
        hi.phase_shift_data(10, angle_units='degreeS')
        self.assertEqual(hi.get_phase_shift(requested_angle_units='radians'), np.deg2rad(10))
