"""
src.recognition
---------------
Phase 4: Star pattern recognition via geometric catalog matching.

Modules
-------
catalog_index  : KD-tree indexed star catalog for O(log n) angular queries
pattern_builder: pixel-to-unit-vector conversion and angular pattern construction
pattern_matcher: hybrid geometric star identification with RANSAC outlier rejection
"""
