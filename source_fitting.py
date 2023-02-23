import numpy as np
import nifty8 as ift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp


def get_gaussian_psf(domain, var):
    # FIXME: cleanup -> refactor into get_gaussian_kernel
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
    X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')

    # normalized psf
    log_psf = - (0.5 / var) * (X ** 2 + Y ** 2)
    log_kernel = ift.makeField(domain, log_psf)
    log_kernel = log_kernel # - np.log(log_kernel.exp().integrate().val)
    p = ift.Plot()
    p.add(log_kernel)
    p.add(log_kernel.exp())
    p.output()
    return log_kernel


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

cluster_sources = np.load(
    '/home/jruestig/pro/python/source_fwd/fits/source_catalogue/150_positions.npy',
    allow_pickle=True
).item()

sources = cluster_sources['bacco_34']
for source in sources:
    fig, axes = plt.subplots(1, 2)
    im = axes[0].imshow(source['source'])
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(source['source'], norm=LogNorm())
    plt.colorbar(im, ax=axes[1])
    plt.show()


s = sources[1]['source']

# source model
isspace = ift.RGSpace(s.shape, 0.04)


noise_scale = 0.01
noise = np.random.normal(size=s.shape, scale=noise_scale)

d = s + noise


cf_make_pars = {
    'offset_mean': 0.,
    'offset_std': (1e-1, 1e-4),
}

cf_fluct_pars = {
    # 'target_subdomain': isspace,
    'scale': (1e-1, 1e-1),
    'cutoff': (1.5, 2e-1),
    'loglogslope': (-4.5, 5e-1)
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


ift.plot_priorsamples(source_diffuse, 9, common_colorbar=False)


dspace = isspace
data = ift.Field.from_raw(dspace, d)

N = ift.ScalingOperator(dspace, noise_scale**2, sampling_dtype=float)

likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ source_diffuse
)

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.1, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(
    name='nonlinear', deltaE=0.5, iteration_limit=5)
nonlinear_sampling = ift.NewtonCG(ic_sampling_nl)
linear_sampling = ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=15)

outputdir = 'output/fullmodel/source_model'


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
