import shgpy
import sympy as sp

fform_filename = 'fform/T_d-S_2-S_2(110)-particularized.p'

fform = shgpy.load_fform(fform_filename)

for pc in fform.get_keys():
    print(pc)
    print('-----------------')
    for i, v in enumerate(fform.get_pc(pc)):
        print(i-16, sp.simplify(sp.expand(v)))
    input('Press enter to continue >>> ')
    
