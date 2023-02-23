import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt


from source_fwd import (
    save_fits, load_fits, Blurring, lens_to_params)

from os.path import join, exists
from image_positions.image_positions import Interpolator

from scipy.interpolate import RectBivariateSpline
from scipy.constants import arcsec



from linear_interpolation import Interpolation, Transponator
from matplotlib.colors import LogNorm

import cluster_fits as cf
from sys import exit

indir = '/home/jruestig/Data/Wolfgang_particles'

parts = [
    'particles_121_p1_main.kappa.fits',
    'particles_140_p1_main.kappa.fits',
    'particles_149_p1_main.kappa.fits',
    'particles_28_p1_main.kappa.fits',
    'particles_99_p1_main.kappa.fits',
]


class Reshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return ift.Field.from_raw(
                self._target, x.val.reshape(self._target.shape)
            )
        else:
            return ift.Field.from_raw(
                self._domain, x.val.reshape(self._domain.shape)
            )



noise_scale = 1.00
detectordis = 0.05

particles, header = load_fits(join(indir, parts[0]), hdr=True)

space = cf.Space(particles.shape, header['CDELT2']/arcsec)

deflection = cf.DeflectionAngle(space)

alpha = deflection(particles)

y = space.xycoords - alpha

S = cf.GaussianSource(space)

x = {'Gauss_0_A': np.array([1.]),
     'Gauss_0_x0': np.array([1.]),
     'Gauss_0_y0': np.array([4.]),
     'Gauss_0_a00': np.array([1.]),
     'Gauss_0_a11': np.array([1.])}

Ls = S.brightness_point(y, x)

plt.imshow(Ls[250:750, 250:750], extent=np.array(space.extent)/2)
plt.show()
