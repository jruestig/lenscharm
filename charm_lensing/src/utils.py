import numpy as np
from astropy.io import fits
from scipy.stats import multivariate_normal


def save_fits(data, name):
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    if name.split('.')[-1] == 'fits':
        hdul.writeto(name, overwrite=True)
    else:
        hdul.writeto(name+'.fits', overwrite=True)


def load_fits(path_to_file, get_header=False):
    with fits.open(path_to_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if get_header:
        return np.array(data).astype(np.float64), header
    return np.array(data).astype(np.float64)


smoother = multivariate_normal.pdf(
    np.array(np.meshgrid(*(np.arange(-10, 10, 1),)*2)).T,
    mean=(0, 0)
)
smoother = smoother/smoother.sum()


def unite_dict(a: dict, b: dict) -> dict:
    '''Returns: union of a and b'''
    tmp = {}
    tmp.update(a)
    tmp.update(b)
    return tmp
