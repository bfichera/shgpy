import unittest

from pathlib import Path
import numpy as np
import sympy as sp
import logging

from shgpy.fformgen import (
    _save_fform_dict,
    _load_fform_dict,
)

_logger = logging.getLogger(__name__)


class TestPickle(unittest.TestCase):

    def test_pickle(self):
        path = Path(__file__).parent / 'fform' / 'test_pickle_fform.p'
        M = 16
        x = sp.symbols('x', real=True)
        iterable = {}
        for k in ['PP', 'PS', 'SP', 'SS']:
            iterable[k] = np.zeros(2*M+1, dtype=sp.Expr)
            for i in np.arange(-16, 17):
                iterable[k][i] = x**2

        _save_fform_dict(path, iterable)
        fform_dict = _load_fform_dict(path)
        path.unlink()
        
        self.assertNotEqual(fform_dict['SS'][0].subs(sp.symbols('x'), 1), 1)
        self.assertEqual(
            fform_dict['SS'][0].subs(sp.symbols('x', real=True), 1),
            1,
        )
        self.assertEqual(fform_dict['SS'][0], x**2)
