import cluster_fits as cf
import numpy as np
from numpy import array
from scipy.interpolate import RectBivariateSpline


def create_mock_data(cfg):
    np.random.seed(cfg['seed'])

    detector_space = cf.Space(
        cfg['spaces']['detector_space']['Npix'],
        cfg['spaces']['detector_space']['distance'])

    dpie = cf.dPIE(detector_space, xy0=np.array((0., 0.)))
    nfw = cf.CircularNfw(detector_space)
    model = dpie + nfw

    lensposition = cfg['mockdata']['Lens']
    cdata = model.convergence_field(lensposition)
    ddata = model.deflection_field(lensposition)

    if cfg['mockdata']['Type'] in ['Gauss', 'GaussianSource']:
        # Gaussian Source
        So = cf.GaussianSource(detector_space)
        spostrue = {'Gauss_0_A': array([20.0]),
                    'Gauss_0_x0': array([0.24]),
                    'Gauss_0_y0': array([0.17]),
                    'Gauss_0_a00': array([0.04]),
                    'Gauss_0_a11': array([0.14])}
        s = So.brightness_point(detector_space.xycoords, spostrue)
        Ls = So.brightness_point(
            detector_space.xycoords-model.deflection_field(lensposition),
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
        s = S(*(detector_space.xycoords), grid=False)*8
        Ls = S(*(detector_space.xycoords-model.deflection_field(lensposition)),
               grid=False)*8

    noise = np.random.normal(size=Ls.shape, scale=cfg['data']['noise_scale'])
    d = Ls + noise

    return s, d, cdata, ddata
