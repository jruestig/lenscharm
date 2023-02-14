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


def get_shift(real, reconstruction, delta):
    # get shift which reduces chi**2 = np.mean((rec-real)**2)
    # always take real source as first entry since rolling gives 0 at edge
    #
    # Note:
    # shiftx = axis=0
    # shifty = axis=1
    res, shiftx, shifty = 1E5, 0, 0
    for ii in np.arange(-delta, delta, 1):
        for jj in np.arange(-delta, delta, 1):
            resi = np.mean(
                (np.roll(np.roll(real, ii, axis=0), jj, axis=1) - reconstruction)**2
            )
            if resi < res:
                res = resi
                shiftx = ii
                shifty = jj
    return res, shiftx, shifty


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

ii = int(sys.argv[-1])

cluster = 'barahir_20'
sources = cluster_sources[cluster]
source = sources[ii]


detectorspace = cf.Space(
    (round(cfg['detector']['fov']/cfg['detector']['resolution']),)*2,
    cfg['detector']['resolution']
)

noise_scale = 0.08
source_resolution = 0.04
resolution = 0.04  # reconstruction

dataname = 'data_0.08_slacs'
lenspath = join(
    '/home/jruestig/pro/python/source_fwd/',
    source[dataname]['path'],
)
position, subsxy0, (mainlenses, _) = lens_to_params(
    join(lenspath, 'lens_{}.data'.format(dataname)))

dpie = cf.dPIE(detectorspace, domainkeys=[ii for ii in range(mainlenses)])
fcnfw = cf.CircularNfw(detectorspace, xy0=subsxy0, fixed=True)
shear = cf.SecondOrderShear(detectorspace, dpie)
model = dpie + shear + fcnfw


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
yslice, xslice = slice(xminarg, xmaxarg), slice(yminarg, ymaxarg)

sim_reconstruction = load_fits(
    join(lenspath, 'best_{}_source.fits'.format(dataname))
)[0]
maxind = np.unravel_index(
    np.argmax(sim_reconstruction, axis=None),
    sim_reconstruction.shape
)
spos = cf.Space(sim_reconstruction.shape, 0.04).xycoords[0, 0, maxind]

if False:
    simulation, clusternumber = cluster.split('_')
    clusternumber = int(clusternumber)
    deflection = cf.load_deflection(
        clusternumber,
        None,
        cfg['cluster'][simulation]['distance'],
        cfg['cluster'][simulation]
    )
    interdeflection = Interpolator(
        cf.Space((1024,)*2, cfg['cluster'][simulation]['distance']),
        deflection
    )
    alpha = interdeflection(detectorspace.xycoords[:, xslice, yslice])
    alpha *= cf.beta_hat(0.4, source['z'])
else:
    alpha = model.deflection_point(
        detectorspace.xycoords[:, xslice, yslice], position)
beta = np.array(detectorspace.xycoords[:, xslice, yslice] - alpha)
# beta = beta[:, ~np.any(beta>50, axis=0)]
# beta = beta[:, ~np.any(beta<-50, axis=0)]
# beta[0] = beta[0] - source['position'][0]
# beta[1] = beta[1] - source['position'][1]
# y = beta[:, source['mask'][xslice, yslice]]
y = beta.reshape(2, -1)


# Defining domains & operators
if False:
    extremum = np.max((np.abs(y.min()), np.abs(y.max()))) + resolution
    sidelength = 2 * extremum
    pixels = int(np.ceil(sidelength/resolution))
    print(pixels)
    y = np.array((y[0]-extremum, y[1]-extremum))  # /(sspace.extent[1]*2)
    isspace = ift.RGSpace((pixels,)*2, resolution)
    interpolator = ift.LinearInterpolator(isspace, array(y))
else:
    y = y - spos.reshape(2, 1)[::-1]
    isspace = ift.RGSpace((512,)*2, resolution)
    ynew = y + np.multiply(isspace.shape, isspace.distances).reshape(2, 1)/2
    interpolator = ift.LinearInterpolator(isspace, array(ynew), cast_to_zero=True)

Trans = Transponator(isspace)


dataname = 'data_{}_{}'.format(
    cfg['detector']['noise'],
    cfg['detector']['path_to_psf'].split('.fits')[0].split('_')[-1]
)
reconstruction_path = join(
    '/home/jruestig/pro/python/source_fwd',
    source[dataname]['path'])
d = load_fits(
    join(reconstruction_path, '{}.fits'.format(dataname))
)
d = d[xslice, yslice]

dspace = ift.RGSpace(d.shape, distances=0.13),
data = ift.makeField(
    ift.UnstructuredDomain(d.reshape(-1).shape),
    d.reshape(-1)
)


psf = load_fits(cfg['detector']['path_to_psf'])[:-1, :]
# B = lambda x: Blurring(x, psf[11:20, 11:20])
B = lambda x: Blurring(x, psf)
BB = ift.JaxLinearOperator(dspace, dspace, B, domain_dtype=float)
BB = ift.DiagonalOperator(
    ift.Field.from_raw(dspace, 1.), sampling_dtype=float)
Re = Reshaper(interpolator.target, dspace)
data = ift.makeField(dspace, d)


def power_spectrum(k):
    return 100/(0.01+k)**4


imargs = {'extent': (xmin, xmax, ymin, ymax), 'origin': 'lower'}
harmonic_space = isspace.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=isspace)
power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = BB @ Re @ interpolator @ Trans @ HT
data_space = R.target
N = ift.ScalingOperator(data_space, noise_scale**2, sampling_dtype=float)

if True:
    print('start rec')
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    IC = ift.GradientNormController(iteration_limit=100, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    m = D(j)

    field = interpolator(Trans(HT(m))).val
    field = B(field.reshape(d.shape))

    source_reconstruction = HT(m).val
    maxi = 120
    maxind = np.unravel_index(
        np.argmax(source_reconstruction, axis=None),
        source_reconstruction.shape
    )

    slicer = slice(maxi//2, (maxi*3)//2)
    # source_real = np.zeros((maxi*2,)*2)
    # source_real[slicer, slicer] = source['source']
    source_real = source['source']
    _, shiftx, shifty = get_shift(
        source_real,
        source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
                              maxind[1]-maxi:maxind[1]+maxi],
        10
    )
    source_reconstruction = np.roll(
        np.roll(source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
                                      maxind[1]-maxi:maxind[1]+maxi],
                -shiftx, axis=0),
        -shifty, axis=1)

    # var = D(ift.full(D.domain, 1.))
    # source_var = HT(var).val
    # source_var = np.roll(
    #     np.roll(source_var[maxind[0]-maxi:maxind[0]+maxi,
    #                        maxind[1]-maxi:maxind[1]+maxi],
    #             -shiftx, axis=0),
    #     -shifty, axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(source['source'], vmin=-0.1, vmax=1.0, origin='lower')
    ims[0, 1] = axes[0, 1].imshow(source_reconstruction, vmin=-0.1, vmax=1.0, origin='lower')
    ims[0, 2] = axes[0, 2].imshow(
        source['source']-source_reconstruction, origin='lower', vmin=-0.3, vmax=0.3, cmap='RdBu_r')
    ims[1, 0] = axes[1, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
    ims[1, 1] = axes[1, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
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

ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=500)
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.05, convergence_level=1, iteration_limit=10)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=50, convergence_level=2)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

N = ift.ScalingOperator(data_space, noise_scale**2, sampling_dtype=float)
likelihood_energy = (
    ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ BB @ Re @ interpolator @ Trans @ correlated_field
)
H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)


try:
    mean = np.load(
        join('/home/jruestig/pro/python/lensing/',
             source[dataname]['path'] + "_KLposition.npy"),
        allow_pickle=True
    ).item()
    mean = ift.MultiField.from_raw(H.domain, mean)
    redo = True
    iteration = 3

    # rec = correlated_field(mean).val
    # source_reconstruction = np.zeros_like(rec)
    # source_reconstruction[rec < 120] = rec[rec < 120]
    # maxi = 120
    # slicer = slice(maxi//2, (maxi*3)//2)
    # source_real = source['source']
    # _, shiftx, shifty = get_shift(
    #     source_real,
    #     source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
    #                           maxind[1]-maxi:maxind[1]+maxi],
    #     10
    # )
    # source_reconstruction = np.roll(
    #     np.roll(source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
    #                                   maxind[1]-maxi:maxind[1]+maxi],
    #             -shiftx, axis=0),
    #     -shifty, axis=1)

    # save_fits(
    #     source_reconstruction,
    #     join('/home/jruestig/pro/python/lensing/',
    #         source[dataname]['path'] + "source.fits")
    # )
    # save_fits(
    #     source_reconstruction,
    #     join('/home/jruestig/pro/python/source_fwd/',
    #         source['data_0.08_slacs']['path'],
    #         'nonlinear_source.fits')
    # )
    # exit()

except FileNotFoundError:
    print('Not previously reconstructed')
    mean = ift.from_random(H.domain, 'normal')
    redo = False
    iteration = 6


N_samples = 5
# Draw new samples to approximate the KL six times
for i in range(iteration):
    if redo:
        i += 5

    if i >= 2:
        if i > 5:
            N_samples = 7

        print(i, 'Samples:{}'.format(N_samples))
        ic_newton = ift.AbsDeltaEnergyController(
            name='Newton {}'.format(i), deltaE=1e-6, convergence_level=1, iteration_limit=20)
        minimizer = ift.NewtonCG(ic_newton)
        KL = ift.SampledKLEnergy(mean, H, N_samples, minimizer_sampling, mirror_samples=True)
    else:
        print(i, 'Samples:{}'.format(N_samples))
        KL = ift.SampledKLEnergy(mean, H, N_samples, None, mirror_samples=True)

    # Draw new samples and minimize KL
    KL, conve = minimizer(KL)
    mean = KL.position

    nmean, var = KL.samples.sample_stat(correlated_field)

    # rec = correlated_field(mean).val
    rec = nmean.val
    source_reconstruction = np.zeros_like(rec)
    source_reconstruction[rec < 120] = rec[rec < 120]
    source_var = var.val

    maxi = 120
    # maxind = np.unravel_index(
    #     np.argmax(source_reconstruction, axis=None),
    #     source_reconstruction.shape
    # )
    slicer = slice(maxi//2, (maxi*3)//2)
    # source_real = np.zeros((maxi*2,)*2)
    # source_real[slicer, slicer] = source['source']
    source_real = source['source']
    _, shiftx, shifty = get_shift(
        source_real,
        source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
                              maxind[1]-maxi:maxind[1]+maxi],
        10
    )
    source_reconstruction = np.roll(
        np.roll(source_reconstruction[maxind[0]-maxi:maxind[0]+maxi,
                                      maxind[1]-maxi:maxind[1]+maxi],
                -shiftx, axis=0),
        -shifty, axis=1)
    # source_var = np.roll(
    #     np.roll(source_var[maxind[0]-maxi:maxind[0]+maxi,
    #                        maxind[1]-maxi:maxind[1]+maxi],
    #             -shiftx, axis=0),
    #     -shifty, axis=1)

    field = BB(Re(interpolator(Trans(nmean)))).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        source['source'], origin='lower', vmin=0.0, vmax=1.0)
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower', vmin=0.0, vmax=1.0)
    ims[0, 2] = axes[0, 2].imshow(
        source['source']-source_reconstruction, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ims[1, 0] = axes[1, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
    ims[1, 1] = axes[1, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
    ims[1, 2] = axes[1, 2].imshow((d-field)/noise_scale, **imargs, cmap='RdBu_r', vmin=-3, vmax=3)
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
    plt.savefig(
            join('/home/jruestig/pro/python/lensing/',
                'output', 'New150sources', cluster, 'tmp',
                '_'.join((
                    str(source['z']),
                    str(source['id_source']),
                    'KL{}.png'.format(i)))
                ))
    plt.close()


np.save(
    join('/home/jruestig/pro/python/lensing/',
         source[dataname]['path'] + "_KLposition"),
    KL.position.val,
    allow_pickle=True
)

save_fits(
    source_reconstruction,
    join('/home/jruestig/pro/python/lensing/',
         source[dataname]['path'] + "source.fits")
)
save_fits(
    source_reconstruction,
    join('/home/jruestig/pro/python/source_fwd/',
         source['data_0.08_slacs']['path'],
         'nonlinear_source.fits')
)
save_fits(
    source['source'],
    join('/home/jruestig/pro/python/lensing/',
         source[dataname]['path'] + "true_source.fits")
)


save_fits(
    (d-field)/noise_scale,
    join('/home/jruestig/pro/python/lensing/',
         source[dataname]['path'] + "residual.fits")
)
save_fits(
    Re(interpolator(Trans(nmean))).val,
    join('/home/jruestig/pro/python/lensing/',
         source[dataname]['path'] + "Ls.fits")
)


fig, axes = plt.subplots(2, 3, figsize=(3.5*4, 2*4))
fig.suptitle('{} z{}_{}    ev={}    X^2={}'.format(
    cluster,
    source['z'],
    source['id_source'],
    0.0,
    (((d-field)/noise_scale)**2).mean()))
ims = np.zeros_like(axes)
ims[0, 0] = axes[0, 0].imshow(
    source['source'], origin='lower', vmin=0.0, vmax=1.0)
ims[0, 1] = axes[0, 1].imshow(
    source_reconstruction, origin='lower', vmin=0.0, vmax=1.0)
ims[0, 2] = axes[0, 2].imshow(
    source['source']-source_reconstruction, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ims[1, 0] = axes[1, 0].imshow(d, **imargs, vmin=-0.10, vmax=0.8)
ims[1, 1] = axes[1, 1].imshow(field, **imargs, vmin=-0.10, vmax=0.8)
ims[1, 2] = axes[1, 2].imshow((d-field)/noise_scale, **imargs, cmap='RdBu_r', vmin=-3, vmax=3)
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
plt.savefig(join(
    '/home/jruestig/pro/python/source_fwd/output/New150sources/check_recs',
    '{}_{}_{}_{}_{}.png'.format(
        cluster, source['z'], source['id_source'], dataname, 'nonlinear')
))
plt.close()
