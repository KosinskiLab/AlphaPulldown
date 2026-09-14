"""Compatibility alias for :mod:`alphapulldown.prediction.fold_preparation`.

Keep module identity so existing imports, patches and pickle globals still work.
New code should import the implementation from ``alphapulldown.prediction``.
"""

import sys

from alphapulldown.prediction import fold_preparation as _implementation

sys.modules[__name__] = _implementation
