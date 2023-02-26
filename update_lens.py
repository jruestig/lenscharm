import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from utils import create_mock_data

from source_fwd import (
    save_fits, load_fits, Blurring, lens_to_params, smoother)
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists
from os import makedirs

from NiftyOperators import PriorTransform

from linear_interpolation import Interpolation, Transponator
from matplotlib.colors import LogNorm

from operators import Reshaper, jax_gaussian
from plotting import deflection_check, Ls_check

import argparse

import cluster_fits as cf
import yaml
import sys
from sys import exit


# TODO:
# - Blurring with psf
# - Update NFW profile (x0, y0, rs)
# - Build in lens-light
# - Find a way to detect sub-structures (update z & b)
# - What could be the correlation structure of the lens-profile
# - Lens shift in Fourier-space: e^(2pi k (x-x0))
# - Shear model
# - Put prior range into mockdata generation and start by some seed
#
# - Try with smaller NFW substructures
# - Try on real data
# - Try with Aleksandra data
#
# DONE:
# - Find the transpose bug (maybe, why is source transposed but rest fine?)


# parser =
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config File", type=str, nargs='?',
                    const=1, default='./configs/first_config.yaml')
args = parser.parse_args()

cfg_file = args.config
with open(cfg_file, 'r') as file:
    cfg = yaml.safe_load(file)


outputdir = cfg['outputdir']
makedirs(outputdir, exist_ok=True)
with open(join(outputdir, cfg_file.split('/')[-1]), 'w') as file:
    yaml.dump(cfg, file)

np.random.seed(41)

noise_scale = 1.00
detectordis = 0.05

detectorspace = cf.Space((128,)*2, detectordis)

if cfg['mock']:
    s, d, cdata, ddata = create_mock_data(cfg)
else:
    d = np.load(cfg['data'])


snrmask = (Blurring(d, smoother) > 2*noise_scale)
SNR = d[snrmask].sum()/(noise_scale*np.sqrt(snrmask.sum()))

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(s, origin='lower')
axes[0, 1].imshow(d, origin='lower')
axes[0, 1].set_title(SNR)
axes[1, 0].imshow(cdata, norm=LogNorm(), origin='lower')
axes[1, 1].imshow(np.hypot(*ddata), origin='lower')
plt.show()


# SPACES
isspace = ift.RGSpace((128,)*2, detectordis)
dspace = ift.RGSpace(d.shape, distances=detectordis)
pointsdomain = ift.UnstructuredDomain(detectorspace.xycoords.reshape(2, -1).shape)


# # SOURCE MODEL
var = ift.LognormalTransform(**cfg['priors']['source']['size']).ducktape_left('size')
G = jax_gaussian(isspace)
Gift = ift.JaxOperator(
    var.target,
    isspace,
    lambda x: G(x['size'])
)
source_mean = Gift @ var

source_maker = ift.CorrelatedFieldMaker('source_')
source_maker.set_amplitude_total_offset(**cfg['priors']['source']['amplitude'])
source_maker.add_fluctuations_matern(isspace, **cfg['priors']['source']['fluctuations'])
source = source_maker.finalize()
source_diffuse = (source_mean + source).exp()


# # Paramatric Source
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

convergence_maker = ift.CorrelatedFieldMaker('lens_')
convergence_maker.set_amplitude_total_offset(**cfg['priors']['lens']['offset'])
convergence_maker.add_fluctuations(isspace, **cfg['priors']['lens']['fluctuations'])
correlated_convergence = convergence_maker.finalize()
convergence_pspec = convergence_maker.power_spectrum

# Mean convergence
cnfw = cf.CircularNfw(detectorspace, xy0=np.array((0, 0)), fixed=True)
nfw_convergence = ift.Field.from_raw(
    isspace, np.log(cnfw.convergence_field(
        {'fCNFW_0_b': np.array(cfg['priors']['lens']['meanconvergence']['fCNFW_0_b']).reshape(1),
         'fCNFW_0_r_s': np.array(cfg['priors']['lens']['meanconvergence']['fCNFW_0_r_s']).reshape(1),
         }))
)
convergence_check = nfw_convergence
nfw_convergence = ift.Adder(nfw_convergence)
adder = nfw_convergence

# Total Convergence
convergence_model = (adder @ correlated_convergence).exp()


# Deflection Angle Converter
tmpdeflection = cf.DeflectionAngle(detectorspace)
deflection = ift.JaxLinearOperator(
    isspace,
    pointsdomain,
    lambda x: tmpdeflection(x).reshape(2, -1),
    domain_dtype=float
)
tmpdeflection = cf.DeflectionAngle(detectorspace)


# Full Lens model
upperleftcorner = np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2
lensmodel = ift.JaxOperator(
    isspace,
    pointsdomain,
    lambda x: (upperleftcorner - isspace.distances[0]/2 +
               (detectorspace.xycoords - tmpdeflection(x)).reshape(2, -1))
)

# FULL MODEL
interpolator = Interpolation(isspace, 'source', pointsdomain, 'lens')
Re = Reshaper(interpolator.target, dspace)

# Try source reconstruction
lens = lensmodel @ convergence_model
fullmodel = Re @ interpolator @ (
    lens.ducktape_left('lens') +
    source_diffuse.ducktape_left('source')
)


if cfg['priorsamples']:
    imargs = {'extent': detectorspace.extent}
    for ii in range(10):
        priorpos = ift.from_random(fullmodel.domain)
        tryer = fullmodel(priorpos)
        source = source_diffuse.force(priorpos)
        conv = convergence_model.force(priorpos)
        defl = deflection(convergence_model.force(priorpos))

        fig, axes = plt.subplots(2, 4)

        im = axes[0, 0].imshow(source.val.T, **imargs)
        plt.colorbar(im , ax=axes[0, 0])
        axes[0, 0].set_title('s')

        im = axes[0, 1].imshow(tryer.val.T, **imargs)
        plt.colorbar(im , ax=axes[0, 1])
        axes[0, 1].set_title('Ls')

        im = axes[0, 2].imshow(d, **imargs)
        plt.colorbar(im, ax=axes[0, 2])
        axes[0, 2].set_title('data')

        im = axes[1, 0].imshow(conv.val.T, **imargs)
        plt.colorbar(im, ax=axes[1, 0])
        axes[1, 0].set_title('Kappa (convergence)')

        im = axes[1, 1].imshow(np.hypot(*defl.val).reshape(128, 128).T,
                            vmax=(np.hypot(*ddata)).max(),
                            **imargs)
        plt.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set_title('alpha (deflectionangle)')

        im = axes[1, 2].imshow(np.hypot(*ddata).reshape(128, 128), **imargs)
        plt.colorbar(im, ax=axes[1, 2])
        axes[1, 2].set_title('alpha_true (deflectionangle)')

        conv = (correlated_convergence.force(priorpos).exp())
        im = axes[0, 3].imshow(conv.val.T, **imargs)
        plt.colorbar(im, ax=axes[0, 3])
        axes[0, 3].set_title('convergence correlated')

        im = axes[1, 3].imshow(
            np.hypot(*(deflection(convergence_check.exp()).val.reshape(2, 128, 128))).T,
            vmax=(np.hypot(*ddata)).max(),
            **imargs)
        plt.colorbar(im, ax=axes[1, 3])
        axes[1, 3].set_title('convergence adder')

        plt.show()


# Data & Likelihood
data = ift.makeField(dspace, d)
N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)
likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ fullmodel
)



# Minimizers
linear_sampling = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_sampling'])
ic_newton = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_newton'])
minimizer = ift.NewtonCG(ic_newton)
if cfg['minimization']['geovi']:
    def nonlinear_sampling(iteration):
        if iteration == 0:
            return None
        else:
            ic_sampling_nl = ift.AbsDeltaEnergyController(
                **cfg['minimization']['ic_sampling_nl'])
            return ift.NewtonCG(ic_sampling_nl)
else:
    nonlinear_sampling = None

def plot_check(samples_list, ii):
    print(f'Plotting iteration {ii} in {outputdir}\n')
    Ls_check(
        samples_list,
        ii,
        outputdir=outputdir,
        source_model=source_diffuse,
        forward_model=fullmodel,
        true_source=s,
        data=d,
        noise_scale=noise_scale,
        extent=detectorspace.extent
    )
    deflection_check(
        samples_list,
        ii,
        outputdir=outputdir,
        convergence_model=convergence_model,
        deflection=deflection,
        deflection_data=ddata,
        convergence_data=cdata,
        extent=detectorspace.extent
    )
    plt.close()



ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.001, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)


samples = ift.optimize_kl(
    likelihood_energy,
    cfg['minimization']['total_iterations'],
    cfg['minimization']['n_samples'],
    minimizer,
    linear_sampling,
    nonlinear_sampling,
    output_directory=outputdir,
    inspect_callback=plot_check,
    initial_position=None,
    # constants=[key for key in sprior.domain.keys()],
    dry_run=False,
)
