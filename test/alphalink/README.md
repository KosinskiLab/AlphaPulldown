This directory contains AlphaLink-specific fixture files.

Active tests now live in the standard buckets:

- `test/unit/test_crosslink_input.py` for the small crosslink helper checks
- `test/cluster/check_alphalink_predictions.py` for GPU and weights-backed smoke tests
- `test/outdated/` for older AlphaLink checks that still hard-code cluster paths or stale fixtures
