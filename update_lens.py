import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt

from utils import create_mock_data
from source_model import source_model

from source_fwd import (
    save_fits, load_fits, Blurring, lens_to_params, smoother)

from os.path import join, exists
from os import makedirs

from functools import partial


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
# - Source-Source reconstruction
# - Volume factor between source and lensplane/detectorspace
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
# - Blurring with psf [DONE]


# parser =
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Config File", type=str, nargs='?',
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

noise_scale = cfg['data']['noise_scale']


# Space convenience
npix_lens = cfg['spaces']['detectorspace']['Npix']
dist_lens = cfg['spaces']['detectorspace']['distance']

npix_source = cfg['spaces']['sourcespace']['Npix']
dist_source = cfg['spaces']['sourcespace']['distance']

detector_space = cf.Space(npix_lens, dist_lens)


if cfg['mock']:
    s, d, c_data, d_data = create_mock_data(cfg)
else:
    d = load_fits(cfg['files']['data_path'])
    s = load_fits(cfg['files']['source_path'])
    psf = load_fits(cfg['files']['psf_path'])

    c_data = np.ones_like(d)
    d_data = np.array((np.ones_like(d),)*2)

snrmask = (Blurring(d, smoother) > 2*noise_scale)
SNR = d[snrmask].sum()/(noise_scale*np.sqrt(snrmask.sum()))

if cfg['data']['data_plot']:
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(s, origin='lower')
    axes[0, 1].imshow(d, origin='lower')
    axes[0, 1].set_title(SNR)
    axes[1, 0].imshow(c_data, norm=LogNorm(), origin='lower')
    axes[1, 1].imshow(np.hypot(*d_data), origin='lower')
    plt.show()


# SPACES
ift_source_space = ift.RGSpace(npix_source, dist_source)
ift_lens_space = ift.RGSpace(npix_lens, dist_lens)
ift_data_space = ift.RGSpace(d.shape, distances=dist_lens)

pointsdomain = ift.UnstructuredDomain(detector_space.xycoords.reshape(2, -1).shape)


# Source
source_dict = source_model(cfg)
source_mean = source_dict['source_mean']
source_matern = source_dict['source_matern']
source_diffuse = source_dict['source_diffuse']

# # Paramatric Source
# spriors = {'Gauss_0_A': ('lognorm', 55., 3.),
#            'Gauss_0_x0': ('normal', 0.0, 0.5),
#            'Gauss_0_y0': ('normal', 0.0, 0.5),
#            'Gauss_0_a00': ('lognorm', .60, 0.3),
#            'Gauss_0_a11': ('lognorm', .60, 0.3)}
# from NiftyOperators import PriorTransform
# sprior = PriorTransform(spriors)
# smodel = ift.JaxOperator(
#     sprior.domain,
#     isspace,
#     lambda x: So.brightness_field(x)
# )
# source_diffuse = smodel @ sprior

convergence_maker = ift.CorrelatedFieldMaker('lens_')
convergence_maker.set_amplitude_total_offset(**cfg['priors']['lens']['offset'])
convergence_maker.add_fluctuations(ift_lens_space, **cfg['priors']['lens']['fluctuations'])
correlated_convergence = convergence_maker.finalize()
convergence_pspec = convergence_maker.power_spectrum

# Mean convergence
cnfw = cf.CircularNfw(detector_space, xy0=np.array((0, 0)), fixed=True)
nfw_convergence = ift.Field.from_raw(
    ift_lens_space, np.log(cnfw.convergence_field(
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
tmpdeflection = cf.DeflectionAngle(detector_space)
deflection = ift.JaxLinearOperator(
    ift_lens_space,
    pointsdomain,
    lambda x: tmpdeflection(x).reshape(2, -1),
    domain_dtype=float
)
tmpdeflection = cf.DeflectionAngle(detector_space)


# Full Lens model
upperleftcorner = np.multiply(
    ift_source_space.shape, ift_source_space.distances).reshape(2, 1)/2
lensmodel = ift.JaxOperator(
    ift_lens_space,
    pointsdomain,
    lambda x: (upperleftcorner - ift_source_space.distances[0]/2 +
               (detector_space.xycoords - tmpdeflection(x)).reshape(2, -1))
)

# FULL MODEL
interpolator = Interpolation(ift_source_space, 'source', pointsdomain, 'lens')
Re = Reshaper(interpolator.target, ift_data_space)


# Psf Blurring
B = partial(Blurring, kernel=psf)
ift_Psf = ift.JaxLinearOperator(ift_data_space, ift_data_space, B, domain_dtype=float)


# Try source reconstruction
lens = lensmodel @ convergence_model
fullmodel = ift_Psf @ Re @ interpolator @ (
    lens.ducktape_left('lens') +
    source_diffuse.ducktape_left('source')
)


if cfg['priorsamples']:
    imargs = {'extent': detector_space.extent}
    for ii in range(10):
        priorpos = ift.from_random(fullmodel.domain)
        print(source_covariance.force(priorpos).val, source_center.force(priorpos).val)
        Ls_prior = fullmodel(priorpos)
        source = source_diffuse.force(priorpos)
        conv = convergence_model.force(priorpos)
        defl = deflection(convergence_model.force(priorpos))

        fig, axes = plt.subplots(3, 3)

        # Source
        im = axes[0, 0].imshow(s, **imargs)
        plt.colorbar(im, ax=axes[0, 0])
        axes[0, 0].set_title('s')

        im = axes[0, 1].imshow(source.val.T, **imargs)
        plt.colorbar(im, ax=axes[0, 1])
        axes[0, 1].set_title('source_prior')

        maternkernel = source_matern.force(priorpos).exp()
        im = axes[0, 2].imshow(maternkernel.val.T, **imargs)
        plt.colorbar(im, ax=axes[0, 2])
        axes[0, 2].set_title('source_matern')

        # Ls
        im = axes[1, 1].imshow(Ls_prior.val.T, **imargs)
        plt.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set_title('Ls_prior')

        im = axes[1, 0].imshow(d, **imargs)
        plt.colorbar(im, ax=axes[1, 0])
        axes[1, 0].set_title('data')

        # Kappa
        im = axes[2, 0].imshow(conv.val.T, **imargs)
        plt.colorbar(im, ax=axes[2, 0])
        axes[2, 0].set_title('Kappa (convergence)')

        im = axes[2, 1].imshow(np.hypot(*defl.val).reshape(*npix_lens).T,
                            # vmax=(np.hypot(*ddata)).max(),
                            **imargs)
        plt.colorbar(im, ax=axes[2, 1])
        axes[2, 1].set_title('alpha (deflectionangle)')

        # im = axes[1, 2].imshow(np.hypot(*ddata).reshape(*dshape), **imargs)
        # plt.colorbar(im, ax=axes[1, 2])
        # axes[1, 2].set_title('alpha_true (deflectionangle)')

        # conv = (correlated_convergence.force(priorpos).exp())
        # im = axes[2, 2].imshow(conv.val.T, **imargs)
        # plt.colorbar(im, ax=axes[2, 2]) #
        # axes[2, 2].set_title('convergence correlated')

        # im = axes[1, 3].imshow(
        #     np.hypot(*(deflection(convergence_check.exp()).val.reshape(2, *npix_lens))).T,
        #     # vmax=(np.hypot(*ddata)).max(),
        #     **imargs)
        # plt.colorbar(im, ax=axes[1, 3])
        # axes[1, 3].set_title('convergence adder')

        plt.show()


# Data & Likelihood
data = ift.makeField(ift_data_space, d)
N = ift.ScalingOperator(ift_data_space, noise_scale**2, sampling_dtype=float)
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
            ic_sampling_nl = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_sampling_nl'])
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
        extent=detector_space.extent,
        samescale=False
    )
    deflection_check(
        samples_list,
        ii,
        outputdir=outputdir,
        convergence_model=convergence_model,
        deflection=deflection,
        deflection_data=d_data,
        convergence_data=c_data,
        extent=detector_space.extent
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
    dry_run=cfg['minimization']['dry_run'],
    resume=cfg['minimization']['resume'],
)
