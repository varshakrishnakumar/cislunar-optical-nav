import os
import sys

_this_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
