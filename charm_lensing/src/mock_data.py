#!/usr/bin/env python3
from typing import Tuple, Any

import cluster_fits as cf
from cluster_fits import Space

import numpy as np
from jax import Array
from numpy import ndarray
from scipy.interpolate import RectBivariateSpline

from charm_lensing.src.spaces import coords
from charm_lensing.src.utils import load_fits


def hubble_deep_field_loader(config: dict)->(np.array, tuple[float]):
    '''Loads an Hubble deep field image'''
    source = np.load(config['hubble_path'], allow_pickle=True).item()
    source = source[int(config['source_id'])]['image']
    source = source/source.max()*config['source_maximum']
    return source, (config['distance'],)*2


def fits_loader(config: dict)->(np.array, tuple[float]):
    '''Loads an fits image'''
    source = load_fits(config['fits_path'])
    source = source/source.max()*config['source_maximum']
    return source, (config['distance'],)*2


class AnalyticSource:
    def __init__(self, A, x0, y0, a00, a11):
        self.source_parameters = {
            'Gauss_0_A': np.array([A]),
            'Gauss_0_x0': np.array([x0]),
            'Gauss_0_y0': np.array([y0]),
            'Gauss_0_a00': np.array([a00]),
            'Gauss_0_a11': np.array([a11]),
        }
        _tmpspace = cf.Space((128,)*2, 0.04)
        self.source_model = cf.GaussianSource(_tmpspace)

    def brightness_point(self, xycoords: np.array) -> np.array:
        return self.source_model.brightness_point(
            xycoords,
            self.source_parameters
        )


class ImageSource:
    def __init__(self, image: np.array, distances: tuple[float]) -> None:
        source_coordinates_0 = coords(image.shape[0], distances[0])
        source_coordinates_1 = coords(image.shape[1], distances[1])
        self.source_model = RectBivariateSpline(
            source_coordinates_0,
            source_coordinates_1,
            image)

    def brightness_point(self, xycoords: np.array) -> np.array:
        return self.source_model(*xycoords, grid=False)


def create_mock_data(
        lens_space: Space,
        source_space: Space,
        noise_scale: float,
        seed: float,
        mock_config: dict,
) -> tuple[Any, int | float | Any, Any, Any]:

    np.random.seed(seed)

    dpie = cf.dPIE(lens_space, xy0=np.array((0., 0.)))
    nfw = cf.CircularNfw(lens_space)
    lens_model = dpie + nfw

    lensposition = mock_config['Lens']
    cdata = lens_model.convergence_field(lensposition)
    ddata = lens_model.deflection_field(lensposition)

    if mock_config['Type'].lower() in ['gauss', 'gaussian']:
        source_evaluater = AnalyticSource(
            A=float(mock_config['gaussian']['A']),
            x0=float(mock_config['gaussian']['x0']),
            y0=float(mock_config['gaussian']['y0']),
            a00=float(mock_config['gaussian']['a00']),
            a11=float(mock_config['gaussian']['a11']),
        )
    elif mock_config['Type'].lower() in ['hubble', 'fits']:
        if mock_config['Type'].lower() in ['hubble']:
            source, distances = hubble_deep_field_loader(mock_config['hubble'])

        elif mock_config['Type'].lower() in ['fits']:
            source, distances = fits_loader(mock_config['fits'])

        source_evaluater = ImageSource(source, distances)

    s = source_evaluater.brightness_point(source_space.xycoords)
    Ls = source_evaluater.brightness_point(
        lens_space.xycoords-lens_model.deflection_field(lensposition))
    noise = np.random.normal(size=Ls.shape, scale=noise_scale)
    d = Ls + noise

    return s, d, cdata, ddata


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt

    from spaces import coords
    from utils import load_fits

    with open('./configs/first_config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']
    npix_source = cfg['spaces']['source_space']['Npix']
    dist_source = cfg['spaces']['source_space']['distance']

    lens_space = cf.Space(npix_lens, dist_lens)
    source_space = cf.Space(npix_source, dist_source)

    noise_scale = cfg['data']['noise_scale']

    s, d, cdata, ddata = create_mock_data(
        lens_space,
        source_space,
        noise_scale,
        cfg['seed'],
        cfg['mock_data'])

    plt.imshow(s, origin='lower')
    plt.show()

    plt.imshow(d, origin='lower')
    plt.show()

    plt.imshow(cdata, origin='lower')
    plt.show()

    plt.imshow(np.hypot(*ddata), origin='lower')
    plt.show()
