import logging
import time

import shgpy
import shgpy.fformgen

mylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

AOI = 0.1745

start = time.time()
mylogger.debug('Starting UFT generation.')
shgpy.fformgen.generate_uncontracted_fourier_transforms(AOI, 'uft/uft10deg')
mylogger.debug(f'Finished UFT generation. Took {time.time()-start} seconds.')
