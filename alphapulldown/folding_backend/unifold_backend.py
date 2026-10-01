"""Compatibility entrypoint for the unavailable legacy UniFold backend.

The packaged ``unifold`` namespace belongs to AlphaLink2. It lacks the old
inference helpers and adds crosslink layers to the network, so routing native
UniFold checkpoints through it is not a supported substitute for UniFold.
"""

from alphapulldown.prediction.inference_flags import validate_backend_availability

from .folding_backend import FoldingBackend


class UnifoldBackend(FoldingBackend):
    """Preserve imports while rejecting legacy calls with an actionable error."""

    @staticmethod
    def setup(*args, **kwargs):
        validate_backend_availability("unifold")

    @staticmethod
    def predict(*args, **kwargs):
        validate_backend_availability("unifold")

    @staticmethod
    def postprocess(*args, **kwargs):
        validate_backend_availability("unifold")
