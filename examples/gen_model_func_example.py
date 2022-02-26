from pathlib import Path
import logging

from shgpy.fformfit import gen_model_func
import shgpy.shg_symbols as S
import shgpy


logging.basicConfig(level=logging.DEBUG)

ROOT_DIR = Path(__file__).parent
SAVE_FOLDER = ROOT_DIR / 'func' / 'Td'
FFORM_FILENAME = ROOT_DIR / 'fform' / 'Td.p'
METHOD = 'clang'

fform = shgpy.load_fform(FFORM_FILENAME)

model_func = gen_model_func(fform, SAVE_FOLDER, method=METHOD)
