import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax import jit

from source_fwd import load_fits, Blurring
from image_positions.image_positions import Interpolator
from scipy.interpolate import RectBivariateSpline
from os.path import join, exists

import cluster_fits as cf
import yaml
import sys
from sys import exit


#TODO:
# - higher source variability  DONE
# - Blurring  - check against no blurring check DONE
# - glamer data  DONE
# - higher SNR ? <- schlechter   DONE
# - correlated field lesen
# put more modular

class Transponator(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # self._check_input(x, mode)
        return ift.makeField(self._domain, x.val.T)


cfg_file = sys.argv[1]
with open(cfg_file, 'r') as file:
    cfg = yaml.safe_load(file)


cluster_sources = np.load(cfg['general']['sources'], allow_pickle=True).item()
clusterlist = list(cluster_sources.keys()); clusterlist.sort()


detectorspace = cf.Space(
    (round(cfg['detector']['fov']/cfg['detector']['resolution']),)*2,
    cfg['detector']['resolution']
)

noise_scale = 0.08
source_resolution = 0.04
resolution = 0.04  # reconstruction

for clusterkey in clusterlist[:1]:
    outputfolder = join(cfg['general']['folder'], 'output', cfg['general']['outputname'], clusterkey)
    simulation, clusternumber = clusterkey.split('_')
    clusternumber = int(clusternumber)

    positions = [source['position'] for source in cluster_sources[clusterkey]]
    redshifts = [source['z'] for source in cluster_sources[clusterkey]]
    sources = []
    for ii in np.argsort(redshifts):
        source = cluster_sources[clusterkey][ii]
        print(source['z'])
        sources.append(
            source['source']/source['source'].max()
        )
    redshifts = np.sort(redshifts)

    convergence, distance, (M200, R200), subs, weights = cf.get_cluster(
        clusternumber, cfg['cluster'][simulation],
        {'zlens': 0.4, 'zsource': 9.0})
    ER = cf.einsteinradius(M200, 0.4, 9.0)
    models, (xy_subs, subs_mas, circulars) = cf.model_creator(
        int(clusterkey.split('_')[1]),
        cfg['reconstruction'],
        ((ER, R200), distance, convergence.shape),
        subs,
        (weights, 0.05),
        subsgetter=True
    )
    recname, model, *_ = models[0]

    path = cfg['reconstruction'][clusterkey.split('_')[0]]
    recposition = np.load(join(path, recname+'.npy'), allow_pickle=True).item()
    masks = []

    intersources = [
        RectBivariateSpline(
            cf.Space(source.shape, source_resolution).xycoords[1, :, 0],
            cf.Space(source.shape, source_resolution).xycoords[0, 0, :],
            source) for source in sources
    ]

    deflection = cf.load_deflection(clusternumber, None, cfg['cluster'][simulation]['distance'], cfg['cluster'][simulation])
    interdeflection = Interpolator(cf.Space((1024,)*2, cfg['cluster'][simulation]['distance']), deflection)

    for ii in range(len(intersources)):
        sourcename = 'source_{}'.format(ii)

        sposition = positions[ii]
        S = intersources[ii]
        sz = redshifts[ii]

        alpha = cf.beta_hat(cfg['cluster'][simulation]['redshift'], sz) * interdeflection(detectorspace.xycoords)
        beta = np.array(detectorspace.xycoords - alpha)
        beta[0] = beta[0] - sposition[0]
        beta[1] = beta[1] - sposition[1]
        # s = S(*(xy-pos for xy, pos in zip(
        #         cf.Space((round(detectorspace.shape[0]*detectorspace.distances[0]/source_resolution),)*2, source_resolution).xycoords,
        #         sposition)),
        #     grid=False)

        mask = cf.Sersic(detectorspace).brightness_point(beta, {'Sersic_0_Ie': 1., 'Sersic_0_Re': 2., 'Sersic_0_n': 3., 'Sersic_0_x0': 0., 'Sersic_0_y0': 0., 'Sersic_0_q': 1., 'Sersic_0_th': 0.}) > cfg['detector']['mask']
        mask = cf.Sersic(detectorspace).brightness_point(beta, {'Sersic_0_Ie': 1., 'Sersic_0_Re': 2., 'Sersic_0_n': 3., 'Sersic_0_x0': 0., 'Sersic_0_y0': 0., 'Sersic_0_q': 1., 'Sersic_0_th': 0.}) > cfg['detector']['mask']*5

        y = beta[:, mask]

        extremum = np.max((np.abs(y.min()), np.abs(y.max()))) + resolution
        sidelength = 2 * extremum
        pixels = int(np.ceil(sidelength/resolution))
        print(pixels)
        space = cf.Space((pixels,)*2, resolution)
        isspace = ift.RGSpace((pixels,)*2, resolution)
        y = np.array((y[0]-extremum, y[1]-extremum))  # /(sspace.extent[1]*2)

        d = load_fits(join(outputfolder, sourcename, 'd_{}.fits'.format(noise_scale)))
        d = load_fits(join(outputfolder, sourcename, 'd_glamer_{}.fits'.format(noise_scale)))
        Ls = S(beta[1], beta[0], grid=False)

        alpha = cf.beta_hat(cfg['cluster'][simulation]['redshift'], sz) * model.deflection_point(detectorspace.xycoords, recposition)
        beta = np.array(detectorspace.xycoords - alpha)
        beta[0] = beta[0] - sposition[0]
        beta[1] = beta[1] - sposition[1]
        y = beta[:, mask]
        y = np.array((y[0]-extremum, y[1]-extremum))  # /(sspace.extent[1]*2)

        break


dspace = detectorspace

interpolator = ift.LinearInterpolator(isspace, array(y.reshape(2, -1)))
Trans = Transponator(isspace)

source = S(*space.xycoords, grid=False)
isource = ift.makeField(isspace, source)


if False:
    def getB(domain, kernel, datamask):
        tmp = np.full(domain.shape, 1)
        arr = np.array(convolve2d(tmp, kernel))
        mask = np.full(arr.shape, False)
        mask[kernel.shape[0]//2:arr.shape[0]-(kernel.shape[0]//2-1),
            kernel.shape[1]//2:arr.shape[1]-(kernel.shape[1]//2-1)] = datamask
        mask = jnp.array(mask)

        def Blurring(field):
            return convolve2d(field, kernel)[mask]

        return Blurring

    zeros = jnp.full(mask.shape, 0.)
    def fielder(x):
        return zeros.at[mask].set(x)

    Blurring = getB(Ls, load_fits(cfg['detector']['path_to_psf'])[:-1, :], mask) #
    B = lambda x: Blurring(fielder(x))

    interpolator = ift.LinearInterpolator(isspace, array(y.reshape(2, -1)))
    Trans = Transponator(isspace)

    source = S(*space.xycoords, grid=False)
    isource = ift.makeField(isspace, source)

    data = ift.makeField(ift.UnstructuredDomain(mask.sum()), d[mask].reshape(-1))
    Bift = ift.JaxLinearOperator(data.domain, data.domain, B, func_T=B)


    def power_spectrum(k):
        return 1000/(0.01+k)**3

    harmonic_space = isspace.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, target=isspace)
    power_space = ift.PowerSpace(harmonic_space)
    PD = ift.PowerDistributor(harmonic_space, power_space)
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    S = ift.DiagonalOperator(prior_correlation_structure)

    R = Bift @ interpolator @ Trans @ HT

    data_space = R.target
    N = ift.ScalingOperator(data_space, noise_scale)
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    m = D(j)

    dspace = detectorspace
    field = np.zeros(dspace.shape)
    field[mask] = interpolator(Trans(HT(m))).val

    imargs = {'extent': detectorspace.extent, 'origin': 'lower'}
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
    ims[0, 1] = axes[0, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
    ims[0, 2] = axes[0, 2].imshow(d-field, **imargs, cmap='RdBu_r')
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
    plt.savefig('output/B_WienerFilter_n{}.png'.format(noise_scale))
    plt.close()



imargs = {'extent': dspace.extent, 'origin': 'lower'}
# field = np.zeros(dspace.shape)
# field[mask] = interpolator(Trans(isource)).val
# fig, axes = plt.subplots(1, 3)
# ims = np.zeros_like(axes)
# ims[0] = axes[0].imshow(d, **imargs)
# ims[1] = axes[1].imshow(field, **imargs)
# ims[2] = axes[2].imshow(d-field, **imargs)
# for im, ax in zip(ims.flatten(), axes.flatten()): plt.colorbar(im, ax=ax)
# axes[0].set_title('data')
# axes[1].set_title('Ls interpolated')
# axes[2].set_title('data - Ls interpolated')
# plt.show()



args = {
    'offset_mean': 0,
    'offset_std': (1e-3, 1e-6),
    # Amplitude of field fluctuations
    'fluctuations': (1., 0.8),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    'loglogavgslope': (-3., 1),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    'flexibility': (1, .5),  # 1.0, 0.5
    # How ragged the integrated Wiener process component is
    'asperity': (0.1, 0.4)  # 0.1, 0.5
}

args = {
    'offset_mean': 0,
    'offset_std': (1e-3, 1e-6),
    # Amplitude of field fluctuations
    'fluctuations': (1., 1e-2),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    'loglogavgslope': (-6., 1),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    'flexibility': (1, .5),  # 1.0, 0.5
    # How ragged the integrated Wiener process component is
    'asperity': (0.1, 0.5)  # 0.1, 0.5
}

correlated_field = ift.exp(ift.SimpleCorrelatedField(isspace, **args))


# Bift => comment d
d = Ls + np.random.normal(scale=noise_scale, size=Ls.shape)
data = ift.makeField(ift.UnstructuredDomain(mask.sum()), d[mask].reshape(-1))

ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=500)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.05, convergence_level=1, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=50, convergence_level=2)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

N = ift.ScalingOperator(data.domain, 1./noise_scale, sampling_dtype=float)
likelihood_energy = (ift.GaussianEnergy(data=data, inverse_covariance=N) @ interpolator @ Trans @ correlated_field)
# likelihood_energy = (ift.GaussianEnergy(data=data, inverse_covariance=N) @ Bift @ interpolator @ Trans @ correlated_field)
H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

initial_mean = ift.from_random(H.domain, 'normal')
mean = initial_mean


import time
t = time.time()
N_samples = 5
# Draw new samples to approximate the KL six times
for i in range(10):
    print(i, 'Samples:{}'.format(N_samples))

    if i > 2:
        ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=0.05, convergence_level=1, iteration_limit=20)
    elif i > 5:
        ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=1e-6, convergence_level=1, iteration_limit=30)
    elif i > 7:
        ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=1E-8, convergence_level=1, iteration_limit=30)
    minimizer = ift.NewtonCG(ic_newton)

    if i > 2:
        if i > 5:
            N_samples = 7
        elif i == 9:
            N_samples = 12
        KL = ift.SampledKLEnergy(mean, H, N_samples, minimizer_sampling, mirror_samples=True)
    else:
        KL = ift.SampledKLEnergy(mean, H, N_samples, None, mirror_samples=True)

    # Draw new samples and minimize KL
    KL, conve = minimizer(KL)
    mean = KL.position

    recsource = correlated_field(mean).val
    sfield = np.zeros_like(recsource)
    sfield[recsource < 1] = recsource[recsource < 1]

    field = np.zeros(dspace.shape)
    field[mask] = interpolator(Trans(correlated_field(mean))).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
    ims[0, 1] = axes[0, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
    ims[0, 2] = axes[0, 2].imshow(d-field, **imargs, cmap='RdBu_r')
    ims[1, 0] = axes[1, 0].imshow(source, origin='lower', vmin=0.0, vmax=1.0)
    ims[1, 1] = axes[1, 1].imshow(sfield.T, origin='lower', vmin=0.0, vmax=1.0)
    ims[1, 2] = axes[1, 2].imshow(source-sfield.T, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[0, 0].set_title('data')
    axes[0, 1].set_title('Ls')
    axes[0, 2].set_title('data - Ls')
    axes[1, 0].set_title('source')
    axes[1, 1].set_title('rec')
    axes[1, 2].set_title('source - rec')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('output/smask_n{}_iter{}.png'.format(noise_scale, i))
    plt.close()
    print((time.time()-t)/60)


def power_spectrum(k):
    return 1000/(0.01+k)**3

harmonic_space = correlated_field.target[0].get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=correlated_field.target[0])
power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = interpolator @ Trans @ HT

data_space = R.target
N = ift.ScalingOperator(data_space, noise_scale)
D_inv = R.adjoint @ N.inverse @ R + S.inverse
j = R.adjoint_times(N.inverse_times(data))
IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
m = D(j)

field = np.zeros(dspace.shape)
field[mask] = interpolator(Trans(HT(m))).val

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
ims = np.zeros_like(axes)
ims[0, 0] = axes[0, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
ims[0, 1] = axes[0, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
ims[0, 2] = axes[0, 2].imshow(d-field, **imargs, cmap='RdBu_r')
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
plt.savefig('output/smask_WienerFilter_n{}.png'.format(noise_scale))
plt.close()


# analytical source check
if False:
    SERSIC = cf.Sersic(space)
    sparams = {'Sersic_0_Ie': np.array(.5).reshape(1),
            'Sersic_0_Re': np.array(1.5/6).reshape(1),
            'Sersic_0_n': np.array(1.7).reshape(1),
            'Sersic_0_x0': 0.0,
            'Sersic_0_y0': 0.0,
            'Sersic_0_q': np.array(.6).reshape(1),
            'Sersic_0_th': np.array(2.1).reshape(1)}
    sparams2 = {'Sersic_0_Ie': np.array(.5).reshape(1),
                'Sersic_0_Re': np.array(1.5/6).reshape(1),
                'Sersic_0_n': np.array(1.7).reshape(1),
                'Sersic_0_x0': sposition[0].reshape(1),
                'Sersic_0_y0': sposition[1].reshape(1),
                'Sersic_0_q': np.array(.6).reshape(1),
                'Sersic_0_th': np.array(2.1).reshape(1)}
    source = array(SERSIC.brightness_point(
        cf.Space(isspace.shape, isspace.distances).xycoords, sparams))
    isource = ift.makeField(isspace, source)
    interpolator = ift.LinearInterpolator(isspace, array(y.reshape(2, -1)))
    Trans = Transponator(isspace)
    dspace = detectorspace
    interdef = lambda x: cf.beta_hat(cfg['cluster'][simulation]['redshift'], sz) * interdeflection(x)
    field = np.zeros(dspace.shape)
    field[mask] = interpolator(Trans(isource)).val
    imargs = {'extent': dspace.extent, 'origin': 'lower'}
    analytical = SERSIC.brightness_point(dspace.xycoords-interdef(dspace.xycoords), sparams2)
    fig, axes = plt.subplots(2, 3)
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(analytical, **imargs)
    ims[0, 1] = axes[0, 1].imshow(field, **imargs)
    ims[0, 2] = axes[0, 2].imshow(field-analytical, **imargs)
    ims[1, 0] = axes[1, 0].imshow(interpolator.adjoint(ift.makeField(
        ift.UnstructuredDomain(mask.sum()),
        array(SERSIC.brightness_point(y, sparams)).reshape(-1))).val, origin='lower')
    ims[1, 1] = axes[1, 1].imshow(source, origin='lower')
    ims[1, 2] = axes[1, 2].imshow((field-array(SERSIC.brightness_point(dspace.xycoords-interdef(detectorspace.xycoords), sparams)))/array(SERSIC.brightness_point(dspace.xycoords-interdef(detectorspace.xycoords), sparams)),
                                vmax=.1, vmin=-.1,
                                **imargs)
    for im, ax in zip(ims.flatten(), axes.flatten()): plt.colorbar(im, ax=ax)
    axes[0, 0].set_title('Ls analytical')
    axes[0, 1].set_title('Ls interpolated')
    axes[0, 2].set_title('inter - analy')
    axes[1, 0].set_title('(LBN)^T d')
    axes[1, 1].set_title('source')
    axes[1, 2].set_title('(inter - analy)/analy')
    plt.show()
