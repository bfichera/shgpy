"""This module defines a number of sympy.Symbol objects
which are used throughout `shgpy`. They are used to define
free parameters in suscepitbility tensors (see
:mod:`~shgpy.tensor_definitions`), and to standardize name
conventions for important variables (like `phi` and `theta`,
which are used throughout shgpy as the standard azimuthal and
incidence angles, respectively).

These variables in this module should be treated as protected and not
modified by the user unless absolutely necessary.

"""
import sympy as sp

xxx = sp.symbols('xxx')
""""""
xxy = sp.symbols('xxy')
""""""
xxz = sp.symbols('xxz')
""""""
xyx = sp.symbols('xyx')
""""""
xyy = sp.symbols('xyy')
""""""
xyz = sp.symbols('xyz')
""""""
xzx = sp.symbols('xzx')
""""""
xzy = sp.symbols('xzy')
""""""
xzz = sp.symbols('xzz')
""""""
yxx = sp.symbols('yxx')
""""""
yxy = sp.symbols('yxy')
""""""
yxz = sp.symbols('yxz')
""""""
yyx = sp.symbols('yyx')
""""""
yyy = sp.symbols('yyy')
""""""
yyz = sp.symbols('yyz')
""""""
yzx = sp.symbols('yzx')
""""""
yzy = sp.symbols('yzy')
""""""
yzz = sp.symbols('yzz')
""""""
zxx = sp.symbols('zxx')
""""""
zxy = sp.symbols('zxy')
""""""
zxz = sp.symbols('zxz')
""""""
zyx = sp.symbols('zyx')
""""""
zyy = sp.symbols('zyy')
""""""
zyz = sp.symbols('zyz')
""""""
zzx = sp.symbols('zzx')
""""""
zzy = sp.symbols('zzy')
""""""
zzz = sp.symbols('zzz')
""""""

sxxx = sp.symbols('sxxx')
""""""
sxxy = sp.symbols('sxxy')
""""""
sxxz = sp.symbols('sxxz')
""""""
sxyx = sp.symbols('sxyx')
""""""
sxyy = sp.symbols('sxyy')
""""""
sxyz = sp.symbols('sxyz')
""""""
sxzx = sp.symbols('sxzx')
""""""
sxzy = sp.symbols('sxzy')
""""""
sxzz = sp.symbols('sxzz')
""""""
syxx = sp.symbols('syxx')
""""""
syxy = sp.symbols('syxy')
""""""
syxz = sp.symbols('syxz')
""""""
syyx = sp.symbols('syyx')
""""""
syyy = sp.symbols('syyy')
""""""
syyz = sp.symbols('syyz')
""""""
syzx = sp.symbols('syzx')
""""""
syzy = sp.symbols('syzy')
""""""
syzz = sp.symbols('syzz')
""""""
szxx = sp.symbols('szxx')
""""""
szxy = sp.symbols('szxy')
""""""
szxz = sp.symbols('szxz')
""""""
szyx = sp.symbols('szyx')
""""""
szyy = sp.symbols('szyy')
""""""
szyz = sp.symbols('szyz')
""""""
szzx = sp.symbols('szzx')
""""""
szzy = sp.symbols('szzy')
""""""
szzz = sp.symbols('szzz')
""""""

xxxx = sp.symbols('xxxx')
""""""
xxxy = sp.symbols('xxxy')
""""""
xxxz = sp.symbols('xxxz')
""""""
xxyx = sp.symbols('xxyx')
""""""
xxyy = sp.symbols('xxyy')
""""""
xxyz = sp.symbols('xxyz')
""""""
xxzx = sp.symbols('xxzx')
""""""
xxzy = sp.symbols('xxzy')
""""""
xxzz = sp.symbols('xxzz')
""""""
xyxx = sp.symbols('xyxx')
""""""
xyxy = sp.symbols('xyxy')
""""""
xyxz = sp.symbols('xyxz')
""""""
xyyx = sp.symbols('xyyx')
""""""
xyyy = sp.symbols('xyyy')
""""""
xyyz = sp.symbols('xyyz')
""""""
xyzx = sp.symbols('xyzx')
""""""
xyzy = sp.symbols('xyzy')
""""""
xyzz = sp.symbols('xyzz')
""""""
xzxx = sp.symbols('xzxx')
""""""
xzxy = sp.symbols('xzxy')
""""""
xzxz = sp.symbols('xzxz')
""""""
xzyx = sp.symbols('xzyx')
""""""
xzyy = sp.symbols('xzyy')
""""""
xzyz = sp.symbols('xzyz')
""""""
xzzx = sp.symbols('xzzx')
""""""
xzzy = sp.symbols('xzzy')
""""""
xzzz = sp.symbols('xzzz')
""""""
yxxx = sp.symbols('yxxx')
""""""
yxxy = sp.symbols('yxxy')
""""""
yxxz = sp.symbols('yxxz')
""""""
yxyx = sp.symbols('yxyx')
""""""
yxyy = sp.symbols('yxyy')
""""""
yxyz = sp.symbols('yxyz')
""""""
yxzx = sp.symbols('yxzx')
""""""
yxzy = sp.symbols('yxzy')
""""""
yxzz = sp.symbols('yxzz')
""""""
yyxx = sp.symbols('yyxx')
""""""
yyxy = sp.symbols('yyxy')
""""""
yyxz = sp.symbols('yyxz')
""""""
yyyx = sp.symbols('yyyx')
""""""
yyyy = sp.symbols('yyyy')
""""""
yyyz = sp.symbols('yyyz')
""""""
yyzx = sp.symbols('yyzx')
""""""
yyzy = sp.symbols('yyzy')
""""""
yyzz = sp.symbols('yyzz')
""""""
yzxx = sp.symbols('yzxx')
""""""
yzxy = sp.symbols('yzxy')
""""""
yzxz = sp.symbols('yzxz')
""""""
yzyx = sp.symbols('yzyx')
""""""
yzyy = sp.symbols('yzyy')
""""""
yzyz = sp.symbols('yzyz')
""""""
yzzx = sp.symbols('yzzx')
""""""
yzzy = sp.symbols('yzzy')
""""""
yzzz = sp.symbols('yzzz')
""""""
zxxx = sp.symbols('zxxx')
""""""
zxxy = sp.symbols('zxxy')
""""""
zxxz = sp.symbols('zxxz')
""""""
zxyx = sp.symbols('zxyx')
""""""
zxyy = sp.symbols('zxyy')
""""""
zxyz = sp.symbols('zxyz')
""""""
zxzx = sp.symbols('zxzx')
""""""
zxzy = sp.symbols('zxzy')
""""""
zxzz = sp.symbols('zxzz')
""""""
zyxx = sp.symbols('zyxx')
""""""
zyxy = sp.symbols('zyxy')
""""""
zyxz = sp.symbols('zyxz')
""""""
zyyx = sp.symbols('zyyx')
""""""
zyyy = sp.symbols('zyyy')
""""""
zyyz = sp.symbols('zyyz')
""""""
zyzx = sp.symbols('zyzx')
""""""
zyzy = sp.symbols('zyzy')
""""""
zyzz = sp.symbols('zyzz')
""""""
zzxx = sp.symbols('zzxx')
""""""
zzxy = sp.symbols('zzxy')
""""""
zzxz = sp.symbols('zzxz')
""""""
zzyx = sp.symbols('zzyx')
""""""
zzyy = sp.symbols('zzyy')
""""""
zzyz = sp.symbols('zzyz')
""""""
zzzx = sp.symbols('zzzx')
""""""
zzzy = sp.symbols('zzzy')
""""""
zzzz = sp.symbols('zzzz')
""""""

theta = sp.symbols('theta', real=True)
"""Angle of incidence

real = True
"""
phi = sp.symbols('phi', real=True)
"""Azimuthal angle

real = True
"""
psi = sp.symbols('psi', real=True)
"""Arbitrary phase shift

real = True
"""
Fx = sp.symbols('Fx')
""""""
Fy = sp.symbols('Fy')
""""""
Fz = sp.symbols('Fz')
""""""
