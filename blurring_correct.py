import numpy as np
from jax.scipy.signal import convolve2d
from astropy.io import fits
import matplotlib.pyplot as plt
from math import ceil


def load_fits(path_to_file, hdr=False):
    with fits.open(path_to_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if hdr:
        return np.array(data).astype(np.float64), header
    return np.array(data).astype(np.float64)


def Blurring(field, kernel):
    arr = np.array(convolve2d(field, kernel))
    return arr[kernel.shape[0]//2:arr.shape[0]-(kernel.shape[0]//2-1),
               kernel.shape[1]//2:arr.shape[1]-(kernel.shape[1]//2-1)]


def Blurr(field, kernel):
    arr = convolve2d(field, kernel)
    slicer = (
        slice(kernel.shape[0]//2, arr.shape[0]-(ceil(kernel.shape[0]/2)-1)),
        slice(kernel.shape[1]//2, arr.shape[1]-(ceil(kernel.shape[1]/2)-1))
    )
    return arr[slicer]

psf = load_fits(
    '/home/jruestig/pro/python/source_fwd/fits/psfs/psf_slacs.fits'
)[:-1, :]


field = np.random.normal(size=(128, 128), scale=1.)
ff = Blurring(field, psf)
fff = Blurr(field, psf)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(ff)
axes[1].imshow(fff)
axes[2].imshow(ff-fff)
plt.show()
