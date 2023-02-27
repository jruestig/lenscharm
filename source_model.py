import nifty8 as ift
from operators import jax_gaussian


def source_model(cfg):
    npix_source = cfg['spaces']['source_space']['Npix']
    dist_source = cfg['spaces']['source_space']['distance']
    ift_source_space = ift.RGSpace(npix_source, dist_source)

    source_center_key = cfg['priors']['source']['center']['key']
    source_covariance_key = cfg['priors']['source']['covariance']['key']
    source_off_diagonal_key = cfg['priors']['source']['off_diagonal']['key']
    source_center = ift.NormalTransform(
        **cfg['priors']['source']['center'], N_copies=2).ducktape_left(source_center_key)
    source_covariance = ift.LognormalTransform(
        **cfg['priors']['source']['covariance'], N_copies=2).ducktape_left(source_covariance_key)
    source_off_diagonal = ift.sigmoid(ift.NormalTransform(
        **cfg['priors']['source']['off_diagonal']).ducktape_left(source_off_diagonal_key))

    G = jax_gaussian(ift_source_space)
    Gift = ift.JaxOperator(
        (source_covariance + source_center + source_off_diagonal).target,
        ift_source_space,
        lambda x: G((x[source_center_key], x[source_covariance_key], x[source_off_diagonal_key]))
    )
    source_mean = Gift @ (source_covariance + source_center + source_off_diagonal)

    source_maker = ift.CorrelatedFieldMaker('source_')
    source_maker.set_amplitude_total_offset(**cfg['priors']['source']['amplitude'])
    source_maker.add_fluctuations_matern(ift_source_space, **cfg['priors']['source']['fluctuations'])
    source_matern = source_maker.finalize()
    source_diffuse = (source_mean + source_matern).exp()
    return {'source_mean': source_mean,
            'source_matern': source_matern,
            'source_diffuse': source_diffuse}


if __name__ == '__main__':
    import yaml
    from source_fwd import load_fits
    import matplotlib.pyplot as plt
    import numpy as np


    cfg_file = './configs/simon_birrer_comp.yaml'
    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    s = load_fits(cfg['files']['source_path'])

    # Source
    cfg['spaces']['source_space']['Npix'] = (320,)*2
    cfg['spaces']['source_space']['distance'] = (0.025/2.5,)*2

    source_dict = source_model(cfg)
    source_mean = source_dict['source_mean']
    source_matern = source_dict['source_matern']
    source_diffuse = source_dict['source_diffuse']

    npix_source = cfg['spaces']['source_space']['Npix']
    dist_source = cfg['spaces']['source_space']['distance']
    ift_source_space = ift.RGSpace(npix_source, dist_source)

    dspace = ift_source_space

    noise_scale = 0.01
    d = s+np.random.normal(scale=noise_scale, size=s.shape)
    data = ift.Field.from_raw(dspace, d)
    N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)


    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.1, iteration_limit=10)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(
        name='nonlinear', deltaE=0.5, iteration_limit=5)
    nonlinear_sampling = ift.NewtonCG(ic_sampling_nl)
    linear_sampling = ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=15)


    def plot_check(samples_list, ii):
        mean, var = samples_list.sample_stat(source_diffuse)

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        im0 = axes[0].imshow(data.val, origin='lower')
        im1 = axes[1].imshow(mean.val, origin='lower')
        im2 = axes[2].imshow(
            (data.val-mean.val)/noise_scale,
            origin='lower', cmap='RdBu_r', vmin=-3.0, vmax=3.0
        )
        for im, ax in zip([im0, im1, im2], axes):
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(f'{outputdir}/source_KL_{ii}.png')
        plt.close()


    likelihood_energy = (
        ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ source_diffuse
    )

    outputdir = 'output/fullmodel/source_model/new_source'
    samples = ift.optimize_kl(
        likelihood_energy,
        5,
        5,
        minimizer,
        linear_sampling,
        nonlinear_sampling,
        output_directory=outputdir,
        inspect_callback=plot_check,
        initial_position=None,
        # constants=[key for key in sprior.domain.keys()],
        dry_run=False,
    )
