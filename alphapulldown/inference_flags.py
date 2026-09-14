"""Compatibility alias for :mod:`alphapulldown.prediction.inference_flags`.

Keep module identity so existing imports, patches and pickle globals still work.
New code should import the implementation from ``alphapulldown.prediction``.
"""

import sys

from alphapulldown.prediction import inference_flags as _implementation

sys.modules[__name__] = _implementation
