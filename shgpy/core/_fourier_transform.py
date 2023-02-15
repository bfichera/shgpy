import logging
import time

import sympy as sp

from .. import shg_symbols as S
from ._lookup_table import _lookup_table


_logger = logging.getLogger(__name__)


def n2i(n, M=16):
    """Convert between Fourier index and array index

    Returns the index (0-`2*M+1`) in a Fourier array corresponding to the `nth` Fourier component
    """
    return n+M


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
    if expr.func == sp.Add or S.phi not in expr.free_symbols:
        if S.phi not in expr.free_symbols:
            if n == 0:
                return expr
            return 0
        relevant_codes = {}
        for code in _lookup_table.keys():
            relevant_codes[code] = _lookup_table[code][n2i(n, M=16)]
        mapping = {
            sp.cos(S.phi)**code[0]*sp.sin(S.phi)**code[1]:ftval
            for code, ftval in relevant_codes.items()
        }
        start = time.time()
        args = []
        for arg in expr.args:
            has_phi = []
            no_phi = []
            for a in arg.args:
                if a.has(S.phi):
                    has_phi.append(a)
                else:
                    no_phi.append(a)
            no_phi.append(mapping[sp.Mul(*has_phi)])
            args.append(sp.Mul(*no_phi))
        ans = expr.func(*args)
        _logger.debug(f'Computing n={n} took {time.time()-start} seconds.')
    else:
        _logger.debug(f'Computing term n={n}, only one arg')
        keyterms = _get_keyterms(expr)
        assert len(keyterms) == 1
        keyterm = keyterms[0]
        code = _get_code(keyterm)
        ftval = _lookup_table[code][n2i(n, M=16)]
        if ftval != 0:
            ans = expr / keyterm * ftval
        else:
            ans = 0
    return ans
