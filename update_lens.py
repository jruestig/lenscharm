import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from source_fwd import (
    save_fits, load_fits, Blurring, lens_to_params, smoother)
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists

from NiftyOperators import PriorTransform

from linear_interpolation import Interpolation, Transponator
from matplotlib.colors import LogNorm

import cluster_fits as cf
import yaml
import sys
from sys import exit


# TODO:
# - Blurring with psf
# - Update NFW profile (x0, y0, rs)
# - Find the transpose bug
# - Build in lens-light
# - Find a way to detect sub-structures (update z & b)
# - What could be the correlation structure of the lens-profile
# - Lens verschiebung als
# -
# - Try with smaller NFW substructures
# - Try on real data
# - Try with Aleksandra data


def jax_gaussian(domain):
    dist_x = domain.distances[0]
    dist_y = domain.distances[1]

    # Periodic Boundary conditions
    x_ax = np.arange(domain.shape[0]) * dist_x
    # x_ax = np.minimum(x_ax, domain.shape[0] - x_ax) * dist_x
    y_ax = np.arange(domain.shape[1]) * dist_x
    # y_ax = np.minimum(y_ax, domain.shape[1] - y_ax) * dist_y

    center = (domain.shape[0]//2,)*2
    x_ax -= center[0] * dist_x
    y_ax -= center[1] * dist_y
    X, Y = jnp.meshgrid(x_ax, y_ax, indexing='ij')

    def gaussian(var):
        return - (0.5 / var) * (X ** 2 + Y ** 2)

    # normalized psf
    return gaussian



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


def get_gaussian_psf(domain, var):
    # FIXME: cleanup -> refactor into get_gaussian_kernel
    dist_x = domain.distances[0]
    dist_y = domain.distances[1]

    # Periodic Boundary conditions
    x_ax = np.arange(domain.shape[0])
    # x_ax = np.minimum(x_ax, domain.shape[0] - x_ax) * dist_x
    y_ax = np.arange(domain.shape[1])
    # y_ax = np.minimum(y_ax, domain.shape[1] - y_ax) * dist_y

    center = (domain.shape[0]//2,)*2
    x_ax -= center[0]
    y_ax -= center[1]
    X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')

    var *= domain.scalar_dvol  # ensures that the variance parameter is specified with respect to the

    # normalized psf
    log_psf = - (0.5 / var) * (X ** 2 + Y ** 2)
    log_kernel = ift.makeField(domain, log_psf)
    log_kernel = log_kernel # - np.log(log_kernel.exp().integrate().val)
    # p = ift.Plot()
    # p.add(log_kernel)
    # p.add(log_kernel.exp())
    # p.output()
    return log_kernel


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


nfw = cf.CircularNfw(detectorspace)
nfwpos = {'CNFW_0_b': 1., 'CNFW_0_r_s': 0.3, 'CNFW_0_x0': 0.0, 'CNFW_0_y0': 1.5}
lpostrue.update(nfwpos)
model = dpie + nfw
cdata = model.convergence_field(lpostrue)
ddata = model.deflection_field(lpostrue)


source = np.load(
    '/home/jruestig/pro/python/source_fwd/fits/source_catalogue/150_positions.npy',
    allow_pickle=True
).item()['bacco_34'][0]['source']


S = RectBivariateSpline(
    cf.Space(source.shape, 0.04).xycoords[1, :, 0],
    cf.Space(source.shape, 0.04).xycoords[0, 0, :],
    source)

s = So.brightness_point(detectorspace.xycoords, spostrue)
Ls = So.brightness_point(detectorspace.xycoords - model.deflection_field(lpostrue), spostrue)
s = S(*(detectorspace.xycoords), grid=False)*8
Ls = S(*(detectorspace.xycoords - model.deflection_field(lpostrue)), grid=False)*8


noise = np.random.normal(size=Ls.shape, scale=noise_scale)
d = Ls + noise

snrmask = (Blurring(d, smoother) > 2*noise_scale)
SNR = d[snrmask].sum()/(noise_scale*np.sqrt(snrmask.sum()))

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(s, origin='lower')
axes[0, 1].imshow(Ls+noise, origin='lower')
axes[0, 1].set_title(SNR)
axes[1, 0].imshow(cdata, norm=LogNorm(), origin='lower')
axes[1, 1].imshow(np.hypot(*ddata), origin='lower')
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
tmpdeflection = lambda x: dpie.deflection_point(detectorspace.xycoords, x)
lmodel = ift.JaxOperator(
    lprior.domain,
    pointsdomain,
    lambda x: (upperleftcorner - isspace.distances[0]/2 +
               (detectorspace.xycoords - tmpdeflection(x)).reshape(2, -1))
)


# # SOURCE MODEL
cf_make_pars = {
    'offset_mean': 0.,
    'offset_std': (1e-1, 1e-4),
}

cf_fluct_pars = {
    # 'target_subdomain': isspace,
    'scale': (1e-1, 1e-1),
    'cutoff': (1., 1e-1),
    'loglogslope': (-5.0, 5e-1)
}

var = ift.LognormalTransform(mean=0.1, sigma=0.07, key='source_size', N_copies=1).ducktape_left('size')
G = jax_gaussian(isspace)
Gift = ift.JaxOperator(
    var.target,
    isspace,
    lambda x: G(x['size'])
)
source_mean = Gift @ var

source_maker = ift.CorrelatedFieldMaker('source_')
source_maker.set_amplitude_total_offset(**cf_make_pars)
source_maker.add_fluctuations_matern(isspace, **cf_fluct_pars)
source = source_maker.finalize()
# source_ps = source_maker.power_spectrum
# source_mean = ift.Adder(get_gaussian_psf(isspace, .1))

source_diffuse = (source_mean + source).exp()


# def pow_spec_source(k):
#     return 1e-2/(0.001 + k**3)
# ham_space = isspace.get_default_codomain()
# pow_space = ift.PowerSpace(ham_space)
# pd = ift.PowerDistributor(ham_space, pow_space)
# HT = ift.HarmonicTransformOperator(ham_space, target=isspace)
# correl_source = pd(ift.PS_field(pow_space, pow_spec_source))
# sc = ift.exp(HT(ift.makeOp(correl_source).ducktape('source')))
# diffuse = sc

# # Source
# spriors = {'Gauss_0_A': ('lognorm', 55., 3.),
#            'Gauss_0_x0': ('normal', 0.0, 0.5),
#            'Gauss_0_y0': ('normal', 0.0, 0.5),
#            'Gauss_0_a00': ('lognorm', .60, 0.3),
#            'Gauss_0_a11': ('lognorm', .60, 0.3)}
# sprior = PriorTransform(spriors)
# smodel = ift.JaxOperator(
#     sprior.domain,
#     isspace,
#     lambda x: So.brightness_field(x)
# )
# source_diffuse = smodel @ sprior


# FULL MODEL
interpolator = Interpolation(isspace, 'source', pointsdomain, 'lens')
Re = Reshaper(interpolator.target, dspace)

fullmodel = Re @ interpolator @ (
    lmodel.ducktape_left('lens') @ lprior + source_diffuse.ducktape_left('source')
)

# To data
data = ift.makeField(dspace, d)

N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)

likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ fullmodel
)


global_iterations = 10


# Convergence Model
args = {
    # 'offset_mean': -0.9,
    # 'offset_std': (1e-5, 1e-6),
    'offset_mean': 0,
    'offset_std': (1e-2, 1e-6),
    # Amplitude of field fluctuations
    'fluctuations': (1.0, 5e-2),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    'loglogavgslope': (-6., 0.7),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    'flexibility': (0.5, 1.0),  # 1.0, 0.5
    # How ragged the integrated Wiener process component is
    'asperity': (0.1, 0.5)  # 0.1, 0.5
}
# args = {
#     'offset_mean': 1, # Gaussian
#     'offset_std': (1e-2, 1e-6),
#     # Amplitude of field fluctuations
#     'fluctuations': (2.0, 1e-1),  # 1.0, 1e-2
#     # Exponent of power law power spectrum component
#     'loglogavgslope': (-5., 0.8),  # -6.0, 1
#     # Amplitude of integrated Wiener process power spectrum component
#     'flexibility': (0.5, 1.0),  # 1.0, 0.5
#     # How ragged the integrated Wiener process component is
#     'asperity': (0.1, 0.5)  # 0.1, 0.5
# }
correlated_convergence = ift.SimpleCorrelatedField(isspace, **args)

# Multiplier
convergence_check = get_gaussian_psf(isspace, 4e4)
gaussian_convergence = ift.Adder(convergence_check)
adder = gaussian_convergence

cnfw = cf.CircularNfw(detectorspace, xy0=np.array((0, 0)), fixed=True)
nfw_convergence = ift.Field.from_raw(
    isspace, np.log(cnfw.convergence_field(
        {'fCNFW_0_b': np.array(1.5).reshape(1),
         'fCNFW_0_r_s': np.array(0.5).reshape(1)}))
)
convergence_check = nfw_convergence
nfw_convergence = ift.Adder(nfw_convergence)
adder = nfw_convergence


convergence_model = (adder @ correlated_convergence).exp()


tmpdeflection = cf.DeflectionAngle(detectorspace)
deflection = ift.JaxLinearOperator(
    isspace,
    pointsdomain,
    lambda x: tmpdeflection(x).reshape(2, -1),
    domain_dtype=float
)


tmpdeflection = cf.DeflectionAngle(detectorspace)

lensmodel = ift.JaxOperator(
    isspace,
    pointsdomain,
    lambda x: (upperleftcorner - isspace.distances[0]/2 +
               (detectorspace.xycoords - tmpdeflection(x)).reshape(2, -1))
)


# Try source reconstruction
lens = lensmodel @ convergence_model
fullmodel = Re @ interpolator @ (
    lens.ducktape_left('lens') +
    source_diffuse.ducktape_left('source')
)

# imargs = {'extent': detectorspace.extent}
# for ii in range(10):
#     priorpos = ift.from_random(fullmodel.domain)
#     tryer = fullmodel(priorpos)
#     source = source_diffuse.force(priorpos)
#     conv = convergence_model.force(priorpos)
#     defl = deflection(convergence_model.force(priorpos))

#     fig, axes = plt.subplots(2, 4)

#     im = axes[0, 0].imshow(source.val.T, **imargs)
#     plt.colorbar(im , ax=axes[0, 0])
#     axes[0, 0].set_title('s')

#     im = axes[0, 1].imshow(tryer.val.T, **imargs)
#     plt.colorbar(im , ax=axes[0, 1])
#     axes[0, 1].set_title('Ls')

#     im = axes[0, 2].imshow(data.val, **imargs)
#     plt.colorbar(im, ax=axes[0, 2])
#     axes[0, 2].set_title('data')

#     im = axes[1, 0].imshow(conv.val.T, **imargs)
#     plt.colorbar(im, ax=axes[1, 0])
#     axes[1, 0].set_title('Kappa (convergence)')

#     im = axes[1, 1].imshow(np.hypot(*defl.val).reshape(128, 128).T,
#                            vmax=(np.hypot(*ddata)).max(),
#                            **imargs)
#     plt.colorbar(im, ax=axes[1, 1])
#     axes[1, 1].set_title('alpha (deflectionangle)')

#     im = axes[1, 2].imshow(np.hypot(*ddata).reshape(128, 128), **imargs)
#     plt.colorbar(im, ax=axes[1, 2])
#     axes[1, 2].set_title('alpha_true (deflectionangle)')

#     conv = (correlated_convergence.force(priorpos).exp())
#     im = axes[0, 3].imshow(conv.val.T, **imargs)
#     plt.colorbar(im, ax=axes[0, 3])
#     axes[0, 3].set_title('convergence correlated')

#     im = axes[1, 3].imshow(
#         np.hypot(*(deflection(convergence_check.exp()).val.reshape(2, 128, 128))).T,
#         vmax=(np.hypot(*ddata)).max(),
#         **imargs)
#     plt.colorbar(im, ax=axes[1, 3])
#     axes[1, 3].set_title('convergence adder')

#     plt.show()

# data = fullmodel(priorpostrue)
# data = So.brightness_point(ddata, spostrue)
# data = ift.makeField(dspace, data.val+noise)

N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)
likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ fullmodel
)


N_samples = 5
global_iterations = 10

ic_sampling = ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=40)
ic_sampling_nl = ift.AbsDeltaEnergyController(name='nonlinear', deltaE=0.5, iteration_limit=10)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.1, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)


outputdir = 'output/fullmodel/transpose_fix'


def deflection_check(samples_list, ii):
    mean, var = samples_list.sample_stat()

    convergence = convergence_model.force(mean).val
    deflectionf = deflection(convergence_model.force(mean)).val.reshape(2, *isspace.shape)

    # number_of_levels = 5
    # levels = np.linspace(start=0.1, stop=1, num=number_of_levels)*cdata.max()

    vmax = np.hypot(*ddata).max()
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # imc = axes[0, 0].contour(cdata, colors='red', levels=levels, origin='lower', extent=detectorspace.extent)
    # axes[0, 0].contour(convergence, colors='orange', levels=levels, origin='lower', extent=detectorspace.extent)
    # axes[0, 1].contour(cdata, colors='red', levels=levels, origin='lower', extent=detectorspace.extent)
    # axes[0, 1].contour(convergence, colors='orange', levels=levels, origin='lower', extent=detectorspace.extent)

    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        cdata,
        origin='lower',
        vmin=cdata.min(), vmax=cdata.max(),
        cmap='RdYlBu_r',
        extent=detectorspace.extent)  # norm=LogNorm(vmax=cdata.max(),vmin=cdata.min()))
    ims[0, 1] = axes[0, 1].imshow(
        convergence,
        origin='lower',
        vmax=cdata.max(), vmin=cdata.min(),
        cmap='RdYlBu_r',
        extent=detectorspace.extent)  # norm=LogNorm(vmax=cdata.max(),vmin=cdata.min()))
    ims[0, 2] = axes[0, 2].imshow(
        (cdata-convergence)/cdata.max(), origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3, extent=detectorspace.extent)
    ims[1, 0] = axes[1, 0].imshow(
        np.hypot(*ddata), vmin=-0.10, origin='lower', vmax=vmax, extent=detectorspace.extent)
    ims[1, 1] = axes[1, 1].imshow(
        np.hypot(*deflectionf), vmin=-0.10, origin='lower', vmax=vmax, extent=detectorspace.extent)
    ims[1, 2] = axes[1, 2].imshow(
        np.hypot(*(ddata-deflectionf))/vmax,
        vmin=-0.3, vmax=0.3,
        origin='lower', cmap='RdBu_r', extent=detectorspace.extent)
    axes[0, 0].set_title('convergence')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('(convergence - rec)/maxconvergence')
    axes[1, 0].set_title('deflection')
    axes[1, 1].set_title('reconstruction')
    axes[1, 2].set_title('(deflection - reconstruction)/maxdeflection')
    for ii, (im, ax) in enumerate(zip(ims.flatten(), axes.flatten())):
        cb = plt.colorbar(im, ax=ax)
        # if ii in [0, 1]:
        #     cb.add_lines(imc)

    plt.tight_layout()
    plt.savefig(f'{outputdir}/deflection_KL_{ii}.png')
    plt.close()


def Ls_check(samples_list, ii):
    mean, var = samples_list.sample_stat()

    source_reconstruction = source_diffuse.force(mean).val.T
    source_std = source_diffuse.force(var).sqrt().val.T
    dfield = fullmodel(mean).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        s, origin='lower', vmin=0, vmax=s.max(), extent=detectorspace.extent)
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower', vmin=0, vmax=s.max(), extent=detectorspace.extent)
    ims[0, 2] = axes[0, 2].imshow(
        (s-source_reconstruction)/s.max(),
        origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3, extent=detectorspace.extent)
    ims[1, 0] = axes[1, 0].imshow(
        data.val, vmin=-0.10, origin='lower', vmax=data.val.max(), extent=detectorspace.extent)
    ims[1, 1] = axes[1, 1].imshow(
        dfield, vmin=-0.10, origin='lower', vmax=data.val.max(), extent=detectorspace.extent)
    ims[1, 2] = axes[1, 2].imshow(
        (data.val-dfield)/noise_scale,
        vmin=-3, vmax=3, origin='lower', cmap='RdBu_r', extent=detectorspace.extent)
    axes[0, 0].set_title('source')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('(source - rec)/std')
    axes[1, 0].set_title('data')
    axes[1, 1].set_title('BLs')
    axes[1, 2].set_title('(data - BLs)/noisescale')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{outputdir}/gauss_KL_{ii}.png')
    plt.close()


def plot_check(samples_list, ii):
    Ls_check(samples_list, ii)
    deflection_check(samples_list, ii)


def linear_sampling(iteration):
    return ift.AbsDeltaEnergyController(name='linear', deltaE=0.05, iteration_limit=25)
    # if iteration < 1:
    #     return ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=10)
    # elif iteration <= 2:
    #     return ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=20)
    # else:
    #     return ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=50)


def nonlinear_sampling(iteration):
    if iteration < 1:
        return None
    elif iteration <= 2:
        ic_sampling_nl = ift.AbsDeltaEnergyController(
            name='nonlinear', deltaE=0.5, iteration_limit=10)
        return ift.NewtonCG(ic_sampling_nl)
    else:
        ic_sampling_nl = ift.AbsDeltaEnergyController(
            name='nonlinear', deltaE=0.5, iteration_limit=10)
        return ift.NewtonCG(ic_sampling_nl)

def Nsamples(iteration):
    if iteration < 3:
        return 5
    else:
        return 8


ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.001, iteration_limit=1)
minimizer = ift.NewtonCG(ic_newton)


samples = ift.optimize_kl(
    likelihood_energy,
    global_iterations,
    Nsamples,
    minimizer,
    linear_sampling,
    nonlinear_sampling,
    output_directory=outputdir,
    inspect_callback=plot_check,
    initial_position=None,
    # constants=[key for key in sprior.domain.keys()],
    dry_run=False,
    resume=True,
)
