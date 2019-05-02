import os
import sys

MODULE_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PACKAGE_PATH = os.path.join(MODULE_PATH, '..')
DATA_PATH    = os.path.join(PACKAGE_PATH, 'data')

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
