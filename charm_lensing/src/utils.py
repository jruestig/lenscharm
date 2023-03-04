import cluster_fits as cf
import numpy as np
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits
from scipy.stats import multivariate_normal   # .pdf(xy, 0, distance*kern)/4


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


def create_mock_data(cfg):
    np.random.seed(cfg['seed'])

    lens_space = cf.Space(
        cfg['spaces']['lens_space']['Npix'],
        cfg['spaces']['lens_space']['distance'])

    dpie = cf.dPIE(lens_space, xy0=np.array((0., 0.)))
    nfw = cf.CircularNfw(lens_space)
    model = dpie + nfw

    lensposition = cfg['mockdata']['Lens']
    cdata = model.convergence_field(lensposition)
    ddata = model.deflection_field(lensposition)

    if cfg['mockdata']['Type'] in ['Gauss', 'GaussianSource']:
        # Gaussian Source
        So = cf.GaussianSource(lens_space)
        spostrue = {'Gauss_0_A': np.array([20.0]),
                    'Gauss_0_x0': np.array([0.24]),
                    'Gauss_0_y0': np.array([0.17]),
                    'Gauss_0_a00': np.array([0.04]),
                    'Gauss_0_a11': np.array([0.14])}
        s = So.brightness_point(lens_space.xycoords, spostrue)
        Ls = So.brightness_point(
            lens_space.xycoords-model.deflection_field(lensposition),
            spostrue)

    elif cfg['mockdata']['Type'] in ['Image']:
        # Hubble Deep Field Source
        source = np.load(
            '/home/jruestig/pro/python/source_fwd/fits/source_catalogue/150_positions.npy',
            allow_pickle=True
        ).item()
        source = source[
            cfg['mockdata']['Image']['cluster']
        ][cfg['mockdata']['Image']['source_id']]['source']
        S = RectBivariateSpline(
            cf.Space(source.shape, 0.04).xycoords[1, :, 0],
            cf.Space(source.shape, 0.04).xycoords[0, 0, :],
            source)
        s = S(*(lens_space.xycoords), grid=False)*8
        Ls = S(*(lens_space.xycoords-model.deflection_field(lensposition)),
               grid=False)*8

    noise = np.random.normal(size=Ls.shape, scale=cfg['data']['noise_scale'])
    d = Ls + noise

    return s, d, cdata, ddata


smoother = multivariate_normal.pdf(
    np.array(np.meshgrid(*(np.arange(-10, 10, 1),)*2)).T,
    mean=(0, 0)
)
smoother = smoother/smoother.sum()


big_smoother = multivariate_normal.pdf(
    np.array(np.meshgrid(*(np.arange(-10, 10, 1),)*2)).T,
    mean=(0, 0),
    cov=4
)
big_smoother = big_smoother/big_smoother.sum()


bigger_smoother = multivariate_normal.pdf(
    np.array(np.meshgrid(*(np.arange(-10, 10, 1),)*2)).T,
    mean=(0, 0),
    cov=11
)
bigger_smoother = bigger_smoother/bigger_smoother.sum()
