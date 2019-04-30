import os
import sys

MODULE_PATH = os.path.abspath('..')
PACKAGE_PATH = os.path.abspath('../..')
DATA_PATH   = os.path.join(PACKAGE_PATH, 'data')

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
