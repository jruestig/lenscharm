import argparse
from functools import partial
from os import makedirs
from os.path import join

import cluster_fits as cf
import matplotlib.pyplot as plt
import nifty8 as ift
import numpy as np
import yaml

from charm_lensing.src.convergence_models import get_convergence_model
from charm_lensing.src.linear_interpolation import Interpolation
from charm_lensing.src.mock_data import create_mock_data
from charm_lensing.src.operators import Reshaper
from charm_lensing.src.plotting import deflection_check, Ls_check, prior_samples_plotting
from charm_lensing.src.psf_operator import PsfOperator
from charm_lensing.src.source_model import source_model
from charm_lensing.src.utils import (load_fits, smoother)

if __name__ == '__main__':
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

    np.random.seed(cfg['seed'])
    ift.random.push_sseq_from_seed(cfg['seed'])

    noise_scale = cfg['data']['noise_scale']

    # Space convenience
    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']

    npix_source = cfg['spaces']['source_space']['Npix']
    dist_source = cfg['spaces']['source_space']['distance']

    lens_space = cf.Space(npix_lens, dist_lens)
    source_space = cf.Space(npix_source, dist_source)

    if cfg['mock']:
        s, d, c_data, d_data = create_mock_data(
            lens_space,
            source_space,
            noise_scale,
            cfg['seed'],
            cfg['mock_data'])
    else:
        d = load_fits(cfg['files']['data_path'])
        if cfg['files']['source_path'] is not None:
            s = load_fits(cfg['files']['source_path'])
        else:
            s = np.ones(npix_source)
            c_data = None
            d_data = None

    snrmask = (PsfOperator(d, smoother) > 2 * noise_scale)
    SNR = d[snrmask].sum() / (noise_scale * np.sqrt(snrmask.sum()))

    if cfg['data_plot']:
        from matplotlib.colors import LogNorm

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
    pointsdomain = ift.UnstructuredDomain(lens_space.xycoords.reshape(2, -1).shape)

    # Source Model
    source_dict = source_model(cfg)
    source_diffuse = source_dict['source_diffuse']

    # Convergence Model
    convergence_dict = get_convergence_model(cfg)
    convergence_model = convergence_dict['full_model_convergence']

    # Deflection Angle Converter
    deflection = cf.DeflectionAngle(lens_space)

    # Full Lens model
    upperleftcorner = np.multiply(
        ift_source_space.shape, ift_source_space.distances).reshape(2, 1) / 2
    lens_model = ift.JaxOperator(
        ift_lens_space,
        pointsdomain,
        lambda x: (upperleftcorner - ift_source_space.distances[0] / 2 +
                   (lens_space.xycoords - deflection(x)).reshape(2, -1))
    )

    # FULL MODEL
    interpolator = Interpolation(ift_source_space, 'source', pointsdomain, 'lens')
    Re = Reshaper(interpolator.target, ift_data_space)

    # Psf Operator
    if cfg['files']['psf_path'] is None:
        print('No Psf loaded')
        ift_Psf = ift.ScalingOperator(ift_data_space, 1, sampling_dtype=float)
    else:
        psf = load_fits(cfg['files']['psf_path'])
        B = partial(PsfOperator, kernel=psf)
        ift_Psf = ift.JaxLinearOperator(ift_data_space, ift_data_space, B, domain_dtype=float)

    # Full model
    lens = lens_model @ convergence_model
    full_model = ift_Psf @ Re @ interpolator @ (
            lens.ducktape_left('lens') +
            source_diffuse.ducktape_left('source')
    )

    if cfg['priorsamples']:
        for ii in range(10):
            prior_samples_plotting(
                full_model, convergence_dict, source_dict, s, d, lens_space.extent)

    # Data & Likelihood
    data = ift.makeField(ift_data_space, d)
    N = ift.ScalingOperator(ift_data_space, noise_scale ** 2, sampling_dtype=float)
    likelihood_energy = (
            ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ full_model
    )

    # Minimizers
    linear_sampling = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_sampling'])
    ic_newton = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_newton'])
    minimizer = ift.NewtonCG(ic_newton)
    if cfg['minimization']['geovi']:
        def nonlinear_sampling(iteration):
            if iteration < cfg['minimization']['geovi_start']:
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
            forward_model=full_model,
            true_source=s,
            data=d,
            noise_scale=noise_scale,
            extent=lens_space.extent,
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
            extent=lens_space.extent
        )
        plt.close()


    ic_newton = ift.AbsDeltaEnergyController(**cfg['minimization']['ic_newton'])
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

    if cfg['calculate_elbo']:
        elbo_stats = ift.estimate_evidence_lower_bound(
            ift.StandardHamiltonian(likelihood_energy),
            samples,
            200
        )
