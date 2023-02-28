from math import ceil
from jax.scipy.signal import convolve2d


def Blurring(field, kernel):
    arr = convolve2d(field, kernel)
    slicer = (
        slice(kernel.shape[0]//2, arr.shape[0]-(ceil(kernel.shape[0]/2)-1)),
        slice(kernel.shape[1]//2, arr.shape[1]-(ceil(kernel.shape[1]/2)-1))
    )
    return arr[slicer]
