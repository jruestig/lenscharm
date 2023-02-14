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

from linear_interpolation import Interpolation, Transponator

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

noise_scale = 1.00
detectordis = 0.05

detectorspace = cf.Space((128,)*2, detectordis)


dpie = cf.dPIE(detectorspace, xy0=np.array((0., 0.)))
lpostrue = {'dPIE_0_b': array([2.0044161]),
            'dPIE_0_q': array([1.47265484]),
            'dPIE_0_r_s': array([0.0471554]),
            'dPIE_0_r_c': array([1.85812236]),
            'dPIE_0_th': array([15.55583871]),
            'dPIE_0_x0': array([0.0320614]),
            'dPIE_0_y0': array([0.05448241])}

So = cf.GaussianSource(detectorspace)
spostrue = {'Gauss_0_A': array([20.0]),
            'Gauss_0_x0': array([0.24]),
            'Gauss_0_y0': array([0.17]),
            'Gauss_0_a00': array([0.04]),
            'Gauss_0_a11': array([0.14])}



s = So.brightness_point(detectorspace.xycoords, spostrue)
Ls = So.brightness_point(detectorspace.xycoords - dpie.deflection_field(lpostrue), spostrue)

noise = np.random.normal(size=Ls.shape, scale=noise_scale)
d = Ls + noise

fig, (a, b) = plt.subplots(1, 2)
a.imshow(s)
b.imshow(Ls+noise)
plt.show()


# SPACES
isspace = ift.RGSpace((128,)*2, detectordis)
dspace = ift.RGSpace(d.shape, distances=detectordis)
pointsdomain = ift.UnstructuredDomain(detectorspace.xycoords.reshape(2, -1).shape)


# LENS MODEL
lenspriorsettings = {
    'dPIE_0_b': ('lognorm', 2.0, 3.0),
    # 'dPIE_0_r_s': ('lognorm', 1.0, 2.0),
    # 'dPIE_0_r_c': ('lognorm', 2.0, 2.0),
    'dPIE_0_r_s': ('uniform', 0.0, 5.0),
    'dPIE_0_r_c': ('uniform', 0.0, 10.0),
    'dPIE_0_x0': ('normal', 0.0, 0.1),
    'dPIE_0_y0': ('normal', 0.0, 0.1),
    'dPIE_0_q': ('uniform', 1.0000000001, 2.0),
    'dPIE_0_th': ('normal', 0.0, 20),
}
lprior = PriorTransform(lenspriorsettings)
upperleftcorner = np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2
lmodel = ift.JaxOperator(
    lprior.domain,
    pointsdomain,
    lambda x: (upperleftcorner - isspace.distances[0]/2 +
               (detectorspace.xycoords -
                dpie.deflection_point(detectorspace.xycoords, x)).reshape(2, -1))
)


# SOURCE MODEL
args = {
    'offset_mean': 0,
    'offset_std': (1e-3, 1e-6),
    # Amplitude of field fluctuations
    'fluctuations': (3., 1e-2),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    'loglogavgslope': (-6., 0.4),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    'flexibility': (1, 1.5),  # 1.0, 0.5
    # How ragged the integrated Wiener process component is
    'asperity': (0.1, 0.5)  # 0.1, 0.5
}
diffuse = ift.exp(ift.SimpleCorrelatedField(isspace, **args))

# def pow_spec_source(k):
#     return 1e-2/(0.001 + k**3)
# ham_space = isspace.get_default_codomain()
# pow_space = ift.PowerSpace(ham_space)
# pd = ift.PowerDistributor(ham_space, pow_space)
# HT = ift.HarmonicTransformOperator(ham_space, target=isspace)
# correl_source = pd(ift.PS_field(pow_space, pow_spec_source))
# sc = ift.exp(HT(ift.makeOp(correl_source).ducktape('source')))
# diffuse = sc

So = cf.GaussianSource(detectorspace)
spriors = {'Gauss_0_A': ('lognorm', 1., 2.),
           'Gauss_0_x0': ('normal', 0.0, 0.5),
           'Gauss_0_y0': ('normal', 0.0, 0.5),
           'Gauss_0_a00': ('lognorm', 1.0, 2.0),
           'Gauss_0_a11': ('lognorm', 1.0, 2.0)}
sprior = PriorTransform(spriors)
smodel = ift.JaxOperator(
    sprior.domain,
    isspace,
    lambda x: So.brightness_field(x)
)
diffuse = smodel @ sprior


# FULL MODEL
interpolator = Interpolation(isspace, 'source', pointsdomain, 'lens')
trans = Transponator(isspace)
Re = Reshaper(interpolator.target, dspace)

fullmodel = trans @ Re @ interpolator @ (
    lmodel.ducktape_left('lens') @ lprior + diffuse.ducktape_left('source')
)

# Fit postrue to prior
strue = {}
strue.update(spostrue)
strue.update(lpostrue)
postrue = ift.MultiField.from_raw(fullmodel.domain, strue)
position = ift.full(fullmodel.domain, 0.)
N = ift.ScalingOperator(fullmodel.domain, (1e-8)**2, sampling_dtype=float)
like = ift.GaussianEnergy(data=postrue, inverse_covariance=N.inverse) @ (lprior+sprior)
posterior = ift.EnergyAdapter(
    position, ift.StandardHamiltonian(like))
ic_newton = ift.AbsDeltaEnergyController(deltaE=0.05)
minimizer = ift.L_BFGS(ic_newton)
posterior, _ = minimizer(posterior)
priorpostrue = posterior.position

posreally = (lprior + sprior)(priorpostrue)

for key, val in posreally.val.items():
    print(key, postrue.val[key][0], val[0], postrue.val[key][0]-val[0], sep='\t')


# To data
data = ift.makeField(dspace, d)

data = fullmodel(priorpostrue)
data = ift.makeField(dspace, data.val+noise)


N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)

likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ fullmodel
)


N_samples = 5
global_iterations = 10

ic_sampling = ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=50)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.1, iteration_limit=20)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(name='nonlinear', deltaE=0.5, iteration_limit=10)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)


def plot_check(samples_list, ii):
    mean, var = samples_list.sample_stat()

    for key, val in lprior.force(mean).val.items():
        print(key, postrue.val[key][0], val[0], sep='\t')
    for key, val in sprior.force(mean).val.items():
        print(key, postrue.val[key][0], val[0], sep='\t')

    source_reconstruction = diffuse.force(mean).val
    dfield = fullmodel(mean).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        s, origin='lower')
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower')
    ims[0, 2] = axes[0, 2].imshow(
        s-source_reconstruction, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ims[1, 0] = axes[1, 0].imshow(data.val, vmin=-0.10)
    ims[1, 1] = axes[1, 1].imshow(dfield, vmin=-0.10)
    ims[1, 2] = axes[1, 2].imshow((data.val-dfield)/noise_scale, vmin=-3, vmax=3)
    axes[0, 0].set_title('source')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('source - rec')
    axes[1, 0].set_title('data')
    axes[1, 1].set_title('BLs')
    axes[1, 2].set_title('(data - BLs)/noisescale')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'output/bla/gauss_diff2_KL_{ii}.png')
    plt.close()


def Nsamples(iteration):
    if iteration < 3:
        return 5
    else:
        return 8

samples = ift.optimize_kl(
    likelihood_energy,
    global_iterations,
    Nsamples,
    minimizer,
    ic_sampling,
    minimizer_sampling,
    output_directory='output/bla/second',
    inspect_callback=plot_check,
    dry_run=False,
)

# defltrue = dpie.deflection_field(postrue)
# deflreco = dpie.deflection_field(lprior.force(mean).val)
# f, axs = plt.subplots(1, 3)
# axs[0].imshow(np.hypot(*defltrue))
# axs[1].imshow(np.hypot(*deflreco))
# axs[2].imshow(np.hypot(*(defltrue-deflreco)))
# plt.show()

for key, val in lprior.force(mean).val.items():
    print(key, postrue.val[key][0], val[0], sep='\t')
for key, val in sprior.force(mean).val.items():
    print(key, postrue.val[key][0], val[0], sep='\t')
