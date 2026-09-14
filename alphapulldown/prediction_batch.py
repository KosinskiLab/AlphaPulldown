"""Compatibility alias for :mod:`alphapulldown.prediction.prediction_batch`.

Keep module identity so existing imports, patches and pickle globals still work.
New code should import the implementation from ``alphapulldown.prediction``.
"""

import sys

from alphapulldown.prediction import prediction_batch as _implementation

sys.modules[__name__] = _implementation
