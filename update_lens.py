import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from source_fwd import (
    save_fits, load_fits, Blurring, lens_to_params)
from image_positions.image_positions import Interpolator
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists

from NiftyOperators import PriorTransform


import cluster_fits as cf
import yaml
import sys
from sys import exit


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


np.random.seed(41)

noise_scale = 0.08
detectordis = 0.05

detectorspace = cf.Space((128,)*2, detectordis)


dpie = cf.dPIE(detectorspace, xy0=np.array((0., 0.)))
prior = PriorTransform(dpie.get_priorparams())
postrue = {'dPIE_0_b': array([1.0044161]),
           'dPIE_0_q': array([1.47265484]),
           'dPIE_0_r_s': array([0.0471554]),
           'dPIE_0_r_c': array([1.85812236]),
           'dPIE_0_th': array([15.55583871]),
           'dPIE_0_x0': array([0.0320614]),
           'dPIE_0_y0': array([0.05448241])}


So = cf.Sersic(detectorspace)
p = {'Sersic_0_Ie': array([1.70399303]),
     'Sersic_0_Re': array([1.10703987]),
     'Sersic_0_n': array([4.49824211]),
     'Sersic_0_q': array([1.34991842]),
     'Sersic_0_th': array([-7.04267101]),
     'Sersic_0_x0': array([0.2595124]),
     'Sersic_0_y0': array([0.18254083])}

Ls = So.brightness_point(detectorspace.xycoords - dpie.deflection_field(postrue), p)/100

noise = np.random.normal(size=Ls.shape, scale=noise_scale)
d = Ls + noise

# plt.imshow(Ls+noise)
# plt.show()


pos = {'dPIE_0_b': array([2.0044161]),
       'dPIE_0_q': array([1.47265484]),
       'dPIE_0_r_s': array([0.0471554]),
       'dPIE_0_r_c': array([1.85812236]),
       'dPIE_0_th': array([15.55583871]),
       'dPIE_0_x0': array([0.0320614]),
       'dPIE_0_y0': array([0.05448241])}


alpha = dpie.deflection_point(detectorspace.xycoords, pos)
beta = np.array(detectorspace.xycoords - alpha)
y = beta.reshape(2, -1)

isspace = ift.RGSpace((128,)*2, detectordis)
s = So.brightness_point(cf.Space(isspace.shape, isspace.distances).xycoords, p)/100
ynew = y + np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2
interpolator = ift.LinearInterpolator(isspace, array(ynew), cast_to_zero=True)


dspace = ift.RGSpace(d.shape, distances=detectordis)
data = ift.makeField(
    ift.UnstructuredDomain(d.reshape(-1).shape),
    d.reshape(-1)
)


Re = Reshaper(interpolator.target, dspace)
data = ift.makeField(dspace, d)


def power_spectrum(k):
    return 1e8/(0.01+k)**4


imargs = {'extent': detectorspace.extent, 'origin': 'lower'}
harmonic_space = isspace.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=isspace)
power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = Re @ interpolator @ HT
data_space = R.target
N = ift.ScalingOperator(data_space, noise_scale**2, sampling_dtype=float)

print('start rec')
D_inv = R.adjoint @ N.inverse @ R + S.inverse
j = R.adjoint_times(N.inverse_times(data))
IC = ift.GradientNormController(iteration_limit=100, tol_abs_gradnorm=1e-3)
D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
m = D(j)

field = interpolator((HT(m))).val
field = field.reshape(d.shape)

source_reconstruction = HT(m).val
maxi = 120
maxind = np.unravel_index(
    np.argmax(source_reconstruction, axis=None),
    source_reconstruction.shape
)

slicer = slice(maxi//2, (maxi*3)//2)
source_real = s

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
ims = np.zeros_like(axes)
ims[0, 0] = axes[0, 0].imshow(source_real, vmin=-0.1, origin='lower')
ims[0, 1] = axes[0, 1].imshow(source_reconstruction, vmin=-0.1, origin='lower')
ims[0, 2] = axes[0, 2].imshow(
    source_real-source_reconstruction, origin='lower', vmin=-0.3, vmax=0.3, cmap='RdBu_r')
ims[1, 0] = axes[1, 0].imshow(d, **imargs, vmin=-0.10)
ims[1, 1] = axes[1, 1].imshow(field, **imargs, vmin=-0.10)
ims[1, 2] = axes[1, 2].imshow(
    (d-field)/noise_scale, **imargs, cmap='RdBu_r', vmin=-3.0, vmax=3.0)
axes[0, 0].set_title('source')
axes[0, 1].set_title('rec')
axes[0, 2].set_title('source - rec')
axes[1, 0].set_title('data')
axes[1, 1].set_title('BLs')
axes[1, 2].set_title('(data - BLs)/noisescale')
for im, ax in zip(ims.flatten(), axes.flatten()):
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
