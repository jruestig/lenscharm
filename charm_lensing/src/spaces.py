#!/usr/bin/env python3

import numpy as np


def coords(shape: int, distance: float) -> np.array:
    '''Returns coordinates such that the edge of the array is shape/2*distance'''
    halfside = shape/2 * distance
    return np.linspace(-halfside+distance/2, halfside-distance/2, shape)
