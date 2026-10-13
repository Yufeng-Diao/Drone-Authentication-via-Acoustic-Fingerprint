# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:49:41 2022

@author: Eidos
"""

# Modified for P1: numerical imports must not load the legacy plotting stack.
def __getattr__(name):
    if name == "CMatrix":
        from toolbox.pyeidos_plot import CMatrix
        return CMatrix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
