import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from functools import partial

from source_fwd import load_fits, Blurring
from image_positions.image_positions import Interpolator
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists

from matplotlib.colors import LogNorm

import cluster_fits as cf
import yaml
import sys
from sys import exit


class Transponator(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # self._check_input(x, mode)
        return ift.makeField(self._domain, x.val.T)


redshift = 1.45
sposition = np.array([-3.39658805,  3.39564086])


noise_scale = 0.08
source_resolution = 0.00022568588643953524
# resolution = source_resolution  # reconstruction
resolution = 0.0025
lensresolution = 0.01


source = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/tmp_source.fits')
d = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/glamer_ls_{}arcsec.fits'.format(lensresolution))
deflection_ = np.array((load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphax_clus.fits'),
                        load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphay_clus.fits')))
deflection_ *= cf.toarcsec

d *= 1e11
d = d + np.random.normal(scale=noise_scale, size=d.shape)
source *= 1e13
mask = d > 0

SO = RectBivariateSpline(
    cf.Space(source.shape, source_resolution).xycoords[1, :, 0],
    cf.Space(source.shape, source_resolution).xycoords[0, 0, :],
    source)

detectorspace = cf.Space(d.shape, lensresolution)

clusternumber = 1
convergence, distance, (M200, R200), subs, weights = cf.get_cluster(
    clusternumber,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4},
    {'zlens': 0.4, 'zsource': 9.0})
ER = cf.einsteinradius(M200, 0.4, 9.0)
models, (xy_subs, subs_mas, circulars) = cf.model_creator(
    clusternumber,
    {'adjust': True,
     'bacco': '/home/jruestig/pro/python/cluster_fits/output/masked/',
     'barahir': '/home/jruestig/pro/python/cluster_fits/output/barahir/',
     'mass': {'upcut': ['max'], 'locut': [5]},
     'regions': {'box': False, 'inner': ['weights'], 'outer': [1.5]},
     'models': {'gals': False,
                'halocomponents': [2],
                'elliptical': ['dPIESH'],
                'circular': ['fCNFW'],
                'shear': ['SH']}},
    ((ER, R200), distance, convergence.shape),
    subs,
    (weights, 0.05),
    subsgetter=True
)
deflection = cf.load_deflection(
    clusternumber,
    None,
    1.3008352168393837,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4})
interdeflection = Interpolator(cf.Space((1024,)*2, 1.3008352168393837), deflection)
recname, model, (boxr, boxo) = models[0]
recposition = np.load(
    join('/home/jruestig/pro/python/lensing/output/interbeginning', recname+'.npy'),
    allow_pickle=True
).item()


# real deflection
if False:
    print('Make mask')
    alpha = cf.beta_hat(0.4, redshift) * interdeflection(detectorspace.xycoords)
    beta = np.array(detectorspace.xycoords - alpha)
    beta[0] = beta[0] - sposition[0]
    beta[1] = beta[1] - sposition[1]
    mask = cf.Sersic(detectorspace).brightness_point(beta, {'Sersic_0_Ie': 1., 'Sersic_0_Re': 2., 'Sersic_0_n': 3., 'Sersic_0_x0': 0., 'Sersic_0_y0': 0., 'Sersic_0_q': 1., 'Sersic_0_th': 0.}) > 15.
    np.save(
        join('/home/jruestig/pro/python/lensing/output/interbeginning', 'mask_{}.npy'.format(lensresolution)),
        np.array(mask)
    )
else:
    mask = np.load(join('/home/jruestig/pro/python/lensing/output/interbeginning', 'mask_{}.npy'.format(lensresolution)))

y = detectorspace.xycoords[:, mask] - cf.beta_hat(0.4, redshift) * interdeflection(detectorspace.xycoords[:, mask])
y = y - sposition.repeat(y.shape[1]).reshape(2, -1)
extremum = np.max((np.abs(y.min()), np.abs(y.max()))) + resolution
sidelength = 2 * extremum
pixels = int(np.ceil(sidelength/resolution))
space = cf.Space((pixels,)*2, resolution)
isspace = ift.RGSpace((pixels,)*2, resolution)
y = np.array((y[0]-extremum, y[1]-extremum))  # /(sspace.extent[1]*2)

if False:
    alpham = model.deflection_point(detectorspace.xycoords[:, mask], recposition)
    betam = np.array(detectorspace.xycoords[:, mask] - alpham)
    betam[0] = betam[0] - sposition[0]
    betam[1] = betam[1] - sposition[1]
    ym = np.array((betam[0]-extremum, betam[1]-extremum))  # /(sspace.extent[1]*2)
else:
    alpham = deflection_[:, mask]
    betam = detectorspace.xycoords[:, mask] - alpham
    betam = betam - sposition.repeat(betam.shape[1]).reshape(2, -1)
    ym = np.array((betam[0]-extremum, betam[1]-extremum))  # /(sspace.extent[1]*2)

data = ift.makeField(ift.UnstructuredDomain(mask.sum()), d[mask].reshape(-1))
pixels = source.shape[0]
interpolator = ift.LinearInterpolator(isspace, array(ym.reshape(2, -1)))
Trans = Transponator(isspace)


ssource = SO(*space.xycoords, grid=False)
isource = ift.makeField(isspace, ssource)

data = interpolator(Trans(isource))
data = ift.makeField(
    data.domain,
    data.val + np.random.normal(scale=noise_scale, size=data.shape)
)
# dd = np.zeros(detectorspace.shape)
# dd[mask] = data.val
# plt.imshow(dd, norm=LogNorm())
# plt.show()

def power_spectrum(k):
    return 1000/(0.01+k)**3

imargs = {'extent': detectorspace.extent, 'origin': 'lower'}

harmonic_space = isspace.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=isspace)
power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = interpolator @ Trans @ HT

data_space = R.target
N = ift.ScalingOperator(data_space, noise_scale)
D_inv = R.adjoint @ N.inverse @ R + S.inverse
j = R.adjoint_times(N.inverse_times(data))
IC = ift.GradientNormController(iteration_limit=20000, tol_abs_gradnorm=1e-3)
print("Wiener")
D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
m = D(j)

field = np.zeros(detectorspace.shape)
field[mask] = interpolator(Trans(HT(m))).val
dd = np.zeros(detectorspace.shape)
dd[mask] = data.val

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
ims = np.zeros_like(axes)
# ims[0, 0] = axes[0, 0].imshow(dd, **imargs)
# ims[0, 1] = axes[0, 1].imshow(field, **imargs)
# ims[0, 2] = axes[0, 2].imshow(dd-field, **imargs, cmap='RdBu_r')
# for ax in axes[0]:
#     ax.contour(mask, **imargs, cmap='Greys')
ims[1, 0] = axes[1, 0].imshow(isource.val, origin='lower')
ims[1, 1] = axes[1, 1].imshow(HT(m).val.T, origin='lower')
ims[1, 2] = axes[1, 2].imshow(isource.val-HT(m).val.T, origin='lower', cmap='RdBu_r')
axes[0, 0].set_title('data')
axes[0, 1].set_title('Ls')
axes[0, 2].set_title('data - Ls')
axes[1, 0].set_title('source')
axes[1, 1].set_title('rec')
axes[1, 2].set_title('source - rec')
for im, ax in zip(ims.flatten(), axes.flatten()):
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(
    join('/home/jruestig/pro/python/lensing/output/interbeginning', 'glamer_truedeflection_WienerFilter_{}.pdf'.format(lensresolution))
)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
ims = np.zeros_like(axes)
ims[0, 0] = axes[0, 0].imshow(isource.val, origin='lower')
ims[0, 1] = axes[0, 1].imshow(HT(mn).val, origin='lower')
ims[0, 2] = axes[0, 2].imshow(isource.val-HT(mn).val, origin='lower', cmap='RdBu_r')
ims[1, 0] = axes[1, 0].imshow(isource.val, origin='lower')
ims[1, 1] = axes[1, 1].imshow(HT(m).val.T, origin='lower')
ims[1, 2] = axes[1, 2].imshow(isource.val-HT(m).val.T, origin='lower', cmap='RdBu_r')
plt.show()
