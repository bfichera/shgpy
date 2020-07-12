import unittest

import numpy as np
import sympy as sp
import logging

import shgpy

_logger = logging.getLogger(__name__)


class TestDeepCopyfForm(unittest.TestCase):

    def test_deepcopy_fform(self):
        M = 16
        x = sp.symbols('x', real=True)
        iterable = {}
        for k in ['PP', 'PS', 'SP', 'SS']:
            iterable[k] = np.zeros(2*M+1, dtype=sp.Expr)
            for i in np.arange(-16, 17):
                iterable[k][i] = x**2
        fform = shgpy.fFormContainer(iterable)
        self.assertNotEqual(fform.get_pc('SS')[16].subs(sp.symbols('x'), 1), 1)
        self.assertEqual(
            fform.get_pc('SS')[16].subs(sp.symbols('x', real=True), 1),
            1,
        )
        self.assertEqual(iterable['SS'][16], x**2)


class TestDeepCopyForm(unittest.TestCase):

    def test_deepcopy_form(self):
        x = sp.symbols('x', real=True)
        iterable = {}
        for k in ['PP', 'PS', 'SP', 'SS']:
            iterable[k] = x**2
        form = shgpy.FormContainer(iterable)
        self.assertNotEqual(form.get_pc('SS').subs(sp.symbols('x'), 1), 1)
        self.assertEqual(
            form.get_pc('SS').subs(sp.symbols('x', real=True), 1),
            1,
        )
        self.assertEqual(iterable['SS'], x**2)
