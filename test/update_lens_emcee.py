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


postrue = {'dPIE_0_b': array([1.4044161]),
           'dPIE_0_q': array([1.47265484]),
           'dPIE_0_r_s': array([0.0471554]),
           'dPIE_0_r_c': array([1.85812236]),
           'dPIE_0_th': array([15.55583871]),
           'dPIE_0_x0': array([0.0320614]),
           'dPIE_0_y0': array([0.05448241])}

msourcep = {'Sersic_0_Ie': array([1.70399303]),
            'Sersic_0_Re': array([2.10703987]),
            'Sersic_0_n': array([2.49824211]),
            'Sersic_0_q': array([1.34991842]),
            'Sersic_0_th': array([-7.04267101]),
            'Sersic_0_x0': array([0.2595124]),
            'Sersic_0_y0': array([0.18254083])}


np.random.seed(41)

noise_scale = 0.02
detectordis = 0.05

detectorspace = cf.Space((128,)*2, detectordis)


dpie = cf.dPIE(detectorspace, xy0=np.array((0., 0.)))


So = cf.Sersic(detectorspace)
Ls = So.brightness_point(detectorspace.xycoords - dpie.deflection_field(postrue), msourcep)/100

noise = np.random.normal(size=Ls.shape, scale=noise_scale)
d = Ls + noise

plt.imshow(Ls+noise)
plt.show()


alpha = dpie.deflection_point(detectorspace.xycoords, postrue)
beta = np.array(detectorspace.xycoords - alpha)
y = beta.reshape(2, -1)

isspace = ift.RGSpace((128,)*2, detectordis)
source_real = So.brightness_point(cf.Space(isspace.shape, isspace.distances).xycoords, msourcep)/100
ynew = y + np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2



from linear_interpolation import Interpolation
interpolator = ift.LinearInterpolator(isspace, array(ynew), cast_to_zero=True)

pointsdomain = ift.UnstructuredDomain(ynew.shape)
interpolator = Interpolation(isspace, 'source', pointsdomain, 'lens')

lenspriorsettings = {
    'dPIE_0_b': ('lognorm', 2.0, 3.0),
    'dPIE_0_r_s': ('lognorm', 1.0, 2.0),
    'dPIE_0_r_c': ('lognorm', 2.0, 2.0),
    'dPIE_0_x0': ('normal', 0.0, 0.1),
    'dPIE_0_y0': ('normal', 0.0, 0.1),
    'dPIE_0_q': ('uniform', 1.0000000001, 2.0),
    'dPIE_0_th': ('normal', 0.0, 20),
    # 'spos_0_x0': ('uniform', -0.2, 0.4),
    # 'spos_0_y0': ('uniform', -0.2, 0.4),
}
lensprior = PriorTransform(lenspriorsettings)
for key, val in lensprior(ift.full(lensprior.domain, 0.)).val.items():
    print(key, val)

for ii in range(10):
    tmp = ift.from_random(lensprior.domain)
    print()
    for key, val in lensprior(tmp).val.items():
        print(key, val)

upperleftcorner = np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2
lmodel = ift.JaxOperator(
    lensprior.domain,
    pointsdomain,
    lambda x: (upperleftcorner +  # jnp.array((x['spos_0_x0'], x['spos_0_y0'])).reshape(2, 1) +
               (detectorspace.xycoords -
               dpie.deflection_point(detectorspace.xycoords, x)).reshape(2, -1))
)


args = {
    'offset_mean': 0,
    'offset_std': (1e-3, 1e-6),
    # Amplitude of field fluctuations
    'fluctuations': (3., 1e-2),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    'loglogavgslope': (-6., 0.4),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    'flexibility': (1, .5),  # 1.0, 0.5
    # How ragged the integrated Wiener process component is
    'asperity': (1.1, 2.5)  # 0.1, 0.5
}
diffuse = ift.exp(ift.SimpleCorrelatedField(isspace, **args))


def pow_spec_source(k):
    return 1./(0.01 + k**2)
ham_space = isspace.get_default_codomain()
pow_space = ift.PowerSpace(ham_space)
pd = ift.PowerDistributor(ham_space, pow_space)
HT = ift.HarmonicTransformOperator(ham_space, target=isspace)
correl_source = pd(ift.PS_field(pow_space, pow_spec_source))
sc = ift.exp(HT(ift.makeOp(correl_source).ducktape('source')))
diffuse = sc

lensmodel = interpolator @ (lmodel.ducktape_left('lens') @ lensprior + diffuse.ducktape_left('source'))


for ii in range(5):
    poslens = ift.from_random(lensprior.domain)
    possource = ift.from_random(diffuse.domain)

    lpos = lensprior(poslens).val
    source = diffuse(possource).val
    lensed_source = lensmodel(poslens.unite(possource)).val

    for key, val in lpos.items():
        print(key, val)

    fig, axes = plt.subplots(2, 3)
    im = axes[0, 0].imshow(source)
    plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(lensed_source.reshape((128,)*2))
    plt.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(d)
    im = axes[1, 0].imshow(dpie.convergence_field(lpos))
    plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].imshow(np.hypot(*dpie.deflection_field(lpos)))
    plt.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(So.brightness_point(
        detectorspace.xycoords- dpie.deflection_field(lpos),
        msourcep
    ))
    plt.colorbar(im, ax=axes[1, 2])
    plt.show()


dspace = ift.RGSpace(d.shape, distances=detectordis)
data = ift.makeField(
    ift.UnstructuredDomain(d.reshape(-1).shape),
    d.reshape(-1)
)


Re = Reshaper(interpolator.target, dspace)
data = ift.makeField(dspace, d)


ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=500)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.05, convergence_level=1, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=50, convergence_level=2)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)

likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ Re @ lensmodel
)
H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

mean = ift.from_random(H.domain, 'normal')
mean = ift.full(H.domain, 0.)

N_samples = 5

print('create KL')
for ii in range(5):
    if ii <= 2:
        KL = ift.SampledKLEnergy(mean, H, N_samples, None, mirror_samples=True)
    if ii > 2:
        KL = ift.SampledKLEnergy(mean, H, N_samples, minimizer_sampling, mirror_samples=True)

    KL, conve = minimizer(KL)
    mean = KL.position

    nmean, var = KL.samples.sample_stat(diffuse)

    # rec = correlated_field(mean).val
    rec = nmean.val
    source_reconstruction = np.zeros_like(rec)
    # source_reconstruction[rec < 120] = rec[rec < 120]
    source_reconstruction = rec
    source_var = var.val

    field = (Re(lensmodel((mean)))).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        source_real, origin='lower')
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower', vmax=1.)
    ims[0, 2] = axes[0, 2].imshow(
        source_real-source_reconstruction, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ims[1, 0] = axes[1, 0].imshow(d, vmin=-0.10, vmax=0.8)
    ims[1, 1] = axes[1, 1].imshow(field, vmin=-0.10, vmax=0.8)
    ims[1, 2] = axes[1, 2].imshow((d-field)/noise_scale, vmin=-3, vmax=3)
    # axes[1, 2].contour(field > 0.1, **imargs, level=['white'])
    axes[0, 0].set_title('source')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('source - rec')
    axes[1, 0].set_title('data')
    axes[1, 1].set_title('BLs')
    axes[1, 2].set_title('(data - BLs)/noisescale')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'output/bla/KL_{ii}.png')
    plt.close()


exit()


interpolator = ift.LinearInterpolator(isspace, array(ynew), cast_to_zero=True)

def power_spectrum(k):
    return 1e8/(0.01+k**2)


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
IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-1)
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




exit()

pos = {'dPIE_0_b': array([1.0044161]),
       'dPIE_0_q': array([1.47265484]),
       'dPIE_0_r_s': array([0.0471554]),
       'dPIE_0_r_c': array([1.85812236]),
       'dPIE_0_th': array([15.55583871]),
       'dPIE_0_x0': array([0.0320614]),
       'dPIE_0_y0': array([0.05448241])}


alpha = dpie.deflection_point(detectorspace.xycoords, pos)
beta = np.array(detectorspace.xycoords - alpha)
y = beta.reshape(2, -1)

alphatrue = dpie.deflection_field(postrue) + noise/4


def log_prior(theta):
    b, q, rs, rc, th, x0, y0 = theta
    if (0 < b < 10) and \
       (1 < q < 10) and \
       (rs < rc) and \
       (0. < rs < 20) and \
       (0. < rc < 25) and \
       (-3 < x0 < 3) and \
       (-3 < y0 < 3):
        return 0.
    return -np.inf

@jit
def fun(pos):
    return -0.5 * np.sum((alphatrue - dpie.deflection_field(pos))**2/noise_scale**2) - Ndim*np.log(2*np.pi/noise_scale)


def fun(pos):
    alpha = dpie.deflection_field(pos)
    beta = np.array(detectorspace.xycoords - alpha)
    y = beta.reshape(2, -1)
    interpolator = ift.LinearInterpolator(isspace, array(ynew), cast_to_zero=True)

    R = Re @ interpolator @ HT
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    IC = ift.GradientNormController(iteration_limit=100, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    m = D(j)

    field = interpolator((HT(m))).val
    field = field.reshape(d.shape)

    return -0.5*np.sum((d - field)**2/noise_scale**2) - (128**2)*np.log(2*np.pi/noise_scale)


Ndim = np.prod(alphatrue.shape)
def log_likelihood(theta):
    pos = {'dPIE_0_b': array([theta[0]]),
           'dPIE_0_q': array([theta[1]]),
           'dPIE_0_r_s': array([theta[2]]),
           'dPIE_0_r_c': array([theta[3]]),
           'dPIE_0_th': array([theta[4]]),
           'dPIE_0_x0': array([theta[5]]),
           'dPIE_0_y0': array([theta[6]])}
    return fun(pos)


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

import corner
import emcee

positrue = np.array([val[0] for _, val in postrue.items()])
posis = np.array([val[0] for _, val in pos.items()])

pos = posis + 1e-1 * np.random.randn(14, posis.shape[0])
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability
)
sampler.run_mcmc(pos, 500, progress=True);

flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
print(flat_samples.shape)


labels = ['b', 'q', 'rs', 'rc', 'th', 'x0', 'y0']
fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()


labels = ['b', 'q', 'rs', 'rc', 'th', 'x0', 'y0']
fig = corner.corner(
    flat_samples, labels=labels, truths=positrue
);
plt.show()
