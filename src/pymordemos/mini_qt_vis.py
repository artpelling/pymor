#!/usr/bin/env python3
from pymor.discretizers.builtin import TriaGrid
from pymor.discretizers.builtin.gui.qt import visualize_patch
from pymor.vectorarrays.numpy import NumpyVectorSpace

grid = TriaGrid()
spc = NumpyVectorSpace(grid.size(2))
vec = spc.random(count=10)
# vec = spc.ones(count=10)
visualize_patch(grid, vec, block=True, backend='matplotlib')
