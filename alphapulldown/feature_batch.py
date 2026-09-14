"""Compatibility alias for :mod:`alphapulldown.features.feature_batch`.

Keep module identity so existing imports, patches and pickle globals still work.
New code should import the implementation from ``alphapulldown.features``.
"""

import sys

from alphapulldown.features import feature_batch as _implementation

sys.modules[__name__] = _implementation
