import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from source_fwd import load_fits, Blurring
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists
from functools import partial

import cluster_fits as cf
import yaml
import sys
from sys import exit

from linear_interpolation import Interpolation, Transponator, Reshaper
from NiftyOperators import PriorTransform

#TODO:
# - correlated field lesen
# put more modular



cfg_file = sys.argv[1]
with open(cfg_file, 'r') as file:
    cfg = yaml.safe_load(file)


cluster_sources = np.load(cfg['general']['sources'], allow_pickle=True).item()



# Settings
noise_scale = 0.08
source_resolution = 0.04
resolution = 0.13  # reconstruction

cluster = 'barahir_20'
sources = cluster_sources[cluster]


# Load Model and start point
outputfolder = join(cfg['general']['folder'], 'output', cfg['general']['outputname'], cluster)
simulation, clusternumber = cluster.split('_')[0], int(cluster.split('_')[1])
convergence, distance, (M200, R200), subs, weights = cf.get_cluster(
    clusternumber, cfg['cluster'][simulation],
    {'zlens': 0.4, 'zsource': 9.0})
ER = cf.einsteinradius(M200, 0.4, 9.0)
models, (xy_subs, subs_mas, circulars) = cf.model_creator(
    clusternumber,
    cfg['reconstruction'],
    ((ER, R200), distance, convergence.shape),
    subs,
    (weights, 0.05),
    subsgetter=True
)
recname, model, *_ = models[0]
path = cfg['reconstruction'][simulation]
recposition = np.load(join(path, recname+'.npy'), allow_pickle=True).item()
priorposition = np.load(join(path, recname+'_priorposition.npy'), allow_pickle=True).item()


# Source Load
source = sources[0]
Ls = source['Ls']
d = Ls + np.random.normal(size=Ls.shape, scale=noise_scale)
mask = source['mask']
DlsDs = cf.beta_hat(0.4, source['z'])


# Data cut
detectorspace = cf.Space(
    (round(cfg['detector']['fov']/cfg['detector']['resolution']),)*2,
    cfg['detector']['resolution']
)
(xmax, ymax), (xmin, ymin) = (
    np.max(detectorspace.xycoords[:, source['mask']], axis=1),
    np.min(detectorspace.xycoords[:, source['mask']], axis=1)
)
fov = detectorspace.shape[0]*cfg['detector']['resolution']
xx = np.linspace(
    -fov/2+cfg['detector']['resolution']/2,
    fov/2-cfg['detector']['resolution']/2,
    source['Ls'].shape[0])
xmaxarg, ymaxarg, xminarg, yminarg = [
    np.argmin(np.abs(xx-tmpx)) for tmpx in [xmax, ymax, xmin, ymin]
]
yslice, xslice = slice(xminarg-10, xmaxarg+10), slice(yminarg-10, ymaxarg+10)

d = d[xslice, yslice]


# NiftySpaces
isspace = ift.RGSpace((256,)*2, source_resolution)
idspace = ift.RGSpace(d.shape, resolution)
upperleftcorner = np.multiply(isspace.shape, isspace.distances).reshape(2, 1)
dspace = cf.Space(Ls.shape, resolution)
pointsdomain = ift.UnstructuredDomain(
    dspace.xycoords[:, mask].reshape(2, -1).shape)

# sspace
alpha = model.deflection_point(dspace.xycoords[:, mask], recposition)
beta = dspace.xycoords[:, mask] - DlsDs*alpha


# Operators
BB = ift.JaxLinearOperator(
    idspace,
    idspace,
    partial(Blurring, kernel=load_fits(cfg['detector']['path_to_psf'])[:-1, :]),
    domain_dtype=float)
Trans = Transponator(idspace)
Interpolator = Interpolation(isspace, 'source', pointsdomain, 'lens')
# Re = Reshaper(Interpolator.target, idspace)


tmpmask = mask[xslice, yslice]

class Reshaper(ift.LinearOperator):
    def __init__(self, domain, target, mask):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._mask = mask
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            out = np.zeros(self._target.shape)
            out[self._mask] = x.val
            return ift.Field.from_raw(self._target, out)
        else:
            return ift.Field.from_raw(
                self._domain, x.val[self._mask]
            )

Re = Reshaper(Interpolator.target, idspace, tmpmask)

# LensModel & Operator
lprior = PriorTransform(model.get_priorparams())
lmodel = ift.JaxOperator(
    lprior.domain,
    pointsdomain,
    lambda x: (upperleftcorner - isspace.distances[0]/2 +
               (dspace.xycoords[:, mask] -
                model.deflection_point(
                    detectorspace.xycoords[:, mask], x)
                ).reshape(2, -1))
)

# SourceModel & Operator
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

# Fullmodel
fullmodel = Re @ Interpolator @ (
    lmodel.ducktape_left('lens') @ lprior + diffuse.ducktape_left('source')
)

data = ift.makeField(idspace, d)
N = ift.ScalingOperator(idspace, noise_scale**2, sampling_dtype=float)


likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ fullmodel
)


startpoint = ift.MultiField.from_raw(lprior.domain, priorposition)
startpoint = startpoint.unite(
    ift.full(diffuse.domain, 0. )
    #ift.from_random(diffuse.domain)
    )

plt.imshow(fullmodel(startpoint).val)
plt.show()

N_samples = 5
global_iterations = 10

ic_sampling = ift.AbsDeltaEnergyController(name='linear', deltaE=0.1, iteration_limit=50)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.1, iteration_limit=20)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(name='nonlinear', deltaE=0.5, iteration_limit=10)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

s = source['source']

outdir = f'output/clusterrec/{cluster}'

def plot_check(samples_list, ii):
    mean, var = samples_list.sample_stat()

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
    plt.savefig(f'{outdir}/KL_{ii}')
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
    initial_position=startpoint,
    constants=[key for key in lprior.domain.keys()],
    output_directory=outdir,
    inspect_callback=plot_check,
    dry_run=False,
)
