import logging

import sympy as sp

from .. import shg_symbols as S
from .data_handler import n2i
from ._lookup_table import _lookup_table


_logger = logging.getLogger(__name__)


def _get_keyterms(expr):
    keyterms = []

    def gen(expr):
        for arg in expr.args:
            if S.phi in arg.free_symbols:
                if arg.func == sp.Pow:
                    for i in range(arg.args[1]):
                        yield arg.args[0]
                else:
                    yield arg

    if expr.func == sp.Mul:
        keyterms.append(sp.Mul(*gen(expr)))
    elif expr.func == sp.Add:
        for arg in expr.args:
            keyterms.append(sp.Mul(*gen(arg)))
    else:
        keyterms.append(expr)
    return list(set(keyterms))


def _get_code(mul):
    num_cos = 0
    num_sin = 0
    if mul.func == sp.Mul:
        for arg in mul.args:
            if arg.func == sp.Pow:
                tp = arg.args[0].func
                if tp == sp.sin:
                    num_sin += arg.args[1]
                elif tp == sp.cos:
                    num_cos += arg.args[1]
            else:
                tp = arg.func
                if tp == sp.sin:
                    num_sin += 1
                elif tp == sp.cos:
                    num_cos += 1
    elif mul.func == sp.Pow:
        tp = mul.args[0].func
        if tp == sp.sin:
            num_sin = mul.args[1]
        elif tp == sp.cos:
            num_cos = mul.args[1]
    return num_cos, num_sin


def _fourier_transform(expr, n, M=16):
    ans = 0
    if expr.func == sp.Add or S.phi not in expr.free_symbols:
        if S.phi not in expr.free_symbols:
            if n == 0:
                return expr
            return 0
        for arg in expr.args:
            keyterms = _get_keyterms(arg)
            assert len(keyterms) == 1
            keyterm = keyterms[0]
            code = _get_code(keyterm)
            ftval = _lookup_table[code][n2i(n, M)]
            ans += arg / keyterm * ftval
    else:
        keyterms = _get_keyterms(expr)
        assert len(keyterms) == 1
        keyterm = keyterms[0]
        code = _get_code(keyterm)
        ftval = _lookup_table[code][n+16]
        ans += expr / keyterm * ftval
    return ans