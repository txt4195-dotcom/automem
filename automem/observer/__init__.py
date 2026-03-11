"""Observer module — per-user 67-dimensional belief vector.

The observer vector lives in the same dimension space as edge scores.
It represents "what this person values" and is used during recall to
compute personalized edge weights via hierarchical dot product.
"""
