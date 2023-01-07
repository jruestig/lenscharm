import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import nifty8 as ift
import cluster_fits as cf

from source_fwd import load_fits
from image_positions.image_positions import Interpolator, images

from scipy.interpolate import RectBivariateSpline
from os.path import join
from sys import exit


# TODO:
# - check different source resolution with true deflection angle to see how reconstructions compare to real source
# - observations of glamer-field with nifty
# - reconstructions of individual fields
# - reconstructions with lensing & real deflection
# - reconstructions with lensing & model deflection


def get_pixel(domain, pointings, units=1):
    a = []
    for ii, maxp in enumerate(np.array(domain.distances)*np.array(domain.shape)/2.):
        x = np.linspace(-maxp, maxp, domain.shape[ii])
        la, lb = np.meshgrid(x, pointings.T[ii]/units)
        a.append(np.argmin(abs(la-lb), axis=1))
    return np.array(a).T


class Transponator(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # self._check_input(x, mode)
        return ift.makeField(self._domain, x.val.T)


def downsample(array, factor):
    out = np.zeros(np.array(array.shape)//factor)
    for i in np.arange(array.shape[0]//factor):
        for j in np.arange(array.shape[1]//factor):
            r = i*factor
            l = j*factor
            s, f = r, r+factor
            t, g = l, l+factor
            out[i, j] = np.sum(array[s:f, t:g])
    return out


redshift = 1.45
sposition = np.array([-3.39658805,  3.39564086])


noise_scale = 0.08
source_resolution = 0.00022568588643953524
# resolution = source_resolution  # reconstruction
# resolution = 0.005
resolution = 0.0025
# resolution = source_resolution * 5
lensresolution = 0.005


source = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/tmp_source.fits')
d = load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/glamer_ls_{}arcsec.fits'.format(lensresolution))
deflection_ = np.array((load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphax_clus_{}arcsec.fits'.format(lensresolution)),
                        load_fits('/home/jruestig/Data/Thorsten_project/CII_sim/CII_sim/alphay_clus_{}arcsec.fits'.format(lensresolution))))
deflection_ *= cf.toarcsec


d *= 1e11
source *= 1e11
mask = d > 0

SO = RectBivariateSpline(
    cf.Space(source.shape, source_resolution).xycoords[1, :, 0],
    cf.Space(source.shape, source_resolution).xycoords[0, 0, :],
    source)

sdown = downsample(source, round(resolution/source_resolution))
SO_down = RectBivariateSpline(
    cf.Space(sdown.shape, resolution).xycoords[1, :, 0],
    cf.Space(sdown.shape, resolution).xycoords[0, 0, :],
    sdown)


detectorspace = cf.Space(d.shape, lensresolution)

clusternumber = 1
convergence, distance, (M200, R200), subs, weights = cf.get_cluster(
    clusternumber,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4},
    {'zlens': 0.4, 'zsource': 9.0})
ER = cf.einsteinradius(M200, 0.4, 9.0)
models, (xy_subs, subs_mas, circulars) = cf.model_creator(
    clusternumber,
    {'adjust': True,
     'bacco': '/home/jruestig/pro/python/cluster_fits/output/masked/',
     'barahir': '/home/jruestig/pro/python/cluster_fits/output/barahir/',
     'mass': {'upcut': ['max'], 'locut': [5]},
     'regions': {'box': False, 'inner': ['weights'], 'outer': [1.5]},
     'models': {'gals': False,
                'halocomponents': [2],
                'elliptical': ['dPIESH'],
                'circular': ['fCNFW'],
                'shear': ['SH']}},
    ((ER, R200), distance, convergence.shape),
    subs,
    (weights, 0.05),
    subsgetter=True
)
deflection = cf.load_deflection(
    clusternumber,
    None,
    1.3008352168393837,
    {'simulation': 'bacco',
     'distance': 1.3008352168393837,
     'mapname': 'big_clusters_adj_',
     'path': '/home/jruestig/Data/angulo/tcm_clusters/',
     'redshift': 0.4})
interdeflection = Interpolator(cf.Space((1024,)*2, 1.3008352168393837), deflection)
recname, model, (boxr, boxo) = models[0]
recposition = np.load(
    join('/home/jruestig/pro/python/lensing/output/interbeginning', recname+'.npy'),
    allow_pickle=True
).item()

# real deflection
if False:
    print('Make mask')
    alpha = cf.beta_hat(0.4, redshift) * interdeflection(detectorspace.xycoords)
    beta = np.array(detectorspace.xycoords - alpha)
    beta[0] = beta[0] - sposition[0]
    beta[1] = beta[1] - sposition[1]
    mask = cf.Sersic(detectorspace).brightness_point(beta, {'Sersic_0_Ie': 1., 'Sersic_0_Re': 2., 'Sersic_0_n': 3., 'Sersic_0_x0': 0., 'Sersic_0_y0': 0., 'Sersic_0_q': 1., 'Sersic_0_th': 0.}) > 15.
    np.save(
        join('/home/jruestig/pro/python/lensing/output/interbeginning', 'mask_{}.npy'.format(lensresolution)),
        np.array(mask)
    )
    imag = images(1.3008352168393837, deflection, sposition, 0.4, redshift)
    np.save(
        join('/home/jruestig/pro/python/lensing/output/interbeginning', 'images.npy'),
        np.array(images)
    )
    imagepixels = get_pixel(detectorspace, imag)
else:
    mask = np.load(join('/home/jruestig/pro/python/lensing/output/interbeginning', 'mask_{}.npy'.format(lensresolution)))
    imag = np.load(
        join('/home/jruestig/pro/python/lensing/output/interbeginning', 'images.npy')
    )
    imagepixels = get_pixel(detectorspace, imag)

# mask = np.zeros_like(mask)
# for pix in imagepixels:
#     mask[pix[1]-50:pix[1]+50, pix[0]-50:pix[0]+50] = True

# y = detectorspace.xycoords[:, mask] - cf.beta_hat(0.4, redshift) * interdeflection(detectorspace.xycoords[:, mask])
y = detectorspace.xycoords[:, mask] - deflection_[:, mask]
y = y - sposition.repeat(y.shape[1]).reshape(2, -1)
extremum = np.max((np.abs(y.min()), np.abs(y.max()))) + resolution
sidelength = 2 * extremum
pixels = int(np.ceil(sidelength/resolution))
space = cf.Space((pixels,)*2, resolution)
isspace = ift.RGSpace((pixels,)*2, resolution)
y = np.array((y[0]-extremum, y[1]-extremum))  # /(sspace.extent[1]*2)

tmpsource = SO(*cf.Space((int(np.ceil(sidelength/source_resolution)),)*2, source_resolution).xycoords, grid=False)
ssource = downsample(
    tmpsource,
    int(resolution/source_resolution)
)
tmp = np.zeros(isspace.shape)
tmp[:-1, :-1] = ssource
# tmp = ssource
isource = ift.makeField(isspace, tmp)

interpolator_glamer = ift.LinearInterpolator(isspace, np.array(y.reshape(2, -1)))

if True:
    print('model deflection')
    alpham = model.deflection_point(detectorspace.xycoords[:, mask], recposition)
    betam = np.array(detectorspace.xycoords[:, mask] - alpham)
    betam[0] = betam[0] - sposition[0]
    betam[1] = betam[1] - sposition[1]
    ym = np.array((betam[0]-extremum, betam[1]-extremum))  # /(sspace.extent[1]*2)
else:
    print('real deflection')
    alpham = deflection_[:, mask]
    betam = detectorspace.xycoords[:, mask] - alpham
    betam = betam - sposition.repeat(betam.shape[1]).reshape(2, -1)
    ym = np.array((betam[0]-extremum, betam[1]-extremum))  # /(sspace.extent[1]*2)

interpolator = ift.LinearInterpolator(isspace, np.array(ym.reshape(2, -1)))
Trans = Transponator(isspace)


if False:
    print('separate')
    size = 50
    mask = np.zeros_like(mask)
    for pix in imagepixels:
        mask[pix[1]-size:pix[1]+size, pix[0]-size:pix[0]+size] = True
    alpham = model.deflection_point(detectorspace.xycoords[:, mask], recposition)
    y = detectorspace.xycoords[:, mask] - alpham
    y = y - sposition.repeat(y.shape[1]).reshape(2, -1)

    # Making isspace
    extremum = np.max((np.abs(y.min()), np.abs(y.max()))) + resolution
    sidelength = 1.5 * extremum
    pixels = int(np.ceil(sidelength/resolution))
    space = cf.Space((pixels,)*2, resolution)
    isspace = ift.RGSpace((pixels,)*2, resolution)

    # Making isource
    tmpsource = SO(*cf.Space((int(np.ceil(sidelength/source_resolution)),)*2, source_resolution).xycoords, grid=False)
    ssource = downsample(
        tmpsource,
        int(resolution/source_resolution)
    )
    isource = ift.makeField(isspace, ssource[:-1, :-1])

    datas = []
    interpolators = []
    for pix in imagepixels:
        masknew = np.zeros_like(mask)
        masknew[pix[1]-size:pix[1]+size, pix[0]-size:pix[0]+size] = True
        dd = d[masknew]
        dtemp = ift.makeField(
            ift.UnstructuredDomain(masknew.sum()), d[masknew].reshape(-1))

        # alpham = model.deflection_point(
        #     detectorspace.xycoords[:, masknew], recposition)
        alpham = deflection_[:, masknew]
        betam = np.array(detectorspace.xycoords[:, masknew] - alpham)
        betam = betam - sposition.repeat(betam.shape[1]).reshape(2, -1)
        ym = np.array((betam[0]-extremum, betam[1]-extremum))
        interpolator = ift.LinearInterpolator(isspace, np.array(ym.reshape(2, -1)))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(dd.reshape((100,)*2)+1, norm=LogNorm())
        axes[1].imshow(interpolator(Trans(isource)).val.reshape((100,)*2)+1, norm=LogNorm())
        plt.show()

        interpolators.append(interpolator)
        # datas.append(dtemp)
        datas.append(interpolator(Trans(isource)))

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

    # Correlated Field & last operator
    Trans = Transponator(isspace)
    correlated_field = ift.exp(ift.SimpleCorrelatedField(isspace, **args))

    # Likelihood and Hamiltonian
    N = ift.ScalingOperator(datas[0].domain, 1./noise_scale, sampling_dtype=float)
    likelihood_energy = (ift.GaussianEnergy(data=datas[0], inverse_covariance=N) @ interpolators[0])
    for data, interpolator in zip(datas[1:], interpolators[1:]):
        likelihood_energy += (ift.GaussianEnergy(data=data, inverse_covariance=N) @ interpolator)
    likelihood_energy = likelihood_energy @ Trans @ correlated_field
    ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=500)
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    initial_mean = ift.from_random(H.domain, 'normal')
    mean = initial_mean

    N_samples = 5
    # Draw new samples to approximate the KL six times
    for i in range(10):
        print(i, 'Samples:{}'.format(N_samples))

        if i > 0:
            ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=1e-3, convergence_level=1, iteration_limit=20)
        elif i > 2:
            print('Here')
            ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=1e-6, convergence_level=1, iteration_limit=30)
        elif i > 7:
            ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=1E-9, convergence_level=1, iteration_limit=30)
        minimizer = ift.NewtonCG(ic_newton)

        if i > 2:
            if i > 5:
                N_samples = 7
            elif i == 9:
                N_samples = 9
            ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=50, convergence_level=2)
            minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
            KL = ift.SampledKLEnergy(mean, H, N_samples, minimizer_sampling, mirror_samples=True)
        else:
            KL = ift.SampledKLEnergy(mean, H, N_samples, None, mirror_samples=True)

        # Draw new samples and minimize KL
        KL, conve = minimizer(KL)
        mean = KL.position

        recsource = correlated_field(mean)
        sfield = recsource.val

        for data, interpolator in zip(datas, interpolators):
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(data.val.reshape((size*2,)*2))
            axes[1].imshow(interpolator(Trans(recsource)).val.reshape((size*2,)*2))
            axes[2].imshow(data.val.reshape((size*2,)*2)-interpolator(Trans(recsource)).val.reshape((size*2,)*2))
            plt.show()

        field = np.zeros(detectorspace.shape)
        field[mask] = interpolator(Trans(correlated_field(mean))).val

        imargs = {'extent': detectorspace.extent, 'origin': 'lower'}
        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        ims = np.zeros_like(axes)
        ims[0, 0] = axes[0, 0].imshow(dd, **imargs, vmin=-0.10, vmax=dd.max()*0.01)
        ims[0, 1] = axes[0, 1].imshow(field, **imargs, vmin=-0.10, vmax=dd.max()*0.01)
        ims[0, 2] = axes[0, 2].imshow((dd-field)/noise_scale, **imargs, cmap='RdBu_r', vmin=-5., vmax=5.)
        ims[1, 0] = axes[1, 0].imshow(isource.val.T/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.5))
        ims[1, 1] = axes[1, 1].imshow(sfield/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.5))
        ims[1, 2] = axes[1, 2].imshow((isource.val.T-sfield)/isource.val.max(), origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        axes[0, 0].set_title('data')
        axes[0, 1].set_title('Ls')
        axes[0, 2].set_title('(data - Ls)/noise')
        axes[1, 0].set_title('source')
        axes[1, 1].set_title('rec')
        axes[1, 2].set_title('source - rec')
        for im, ax in zip(ims.flatten(), axes.flatten()):
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(
            join('/home/jruestig/pro/python/lensing/output/interbeginning/KLs',
                'newmask_dglamer_defmodel_lr{}_sr{}_iter{}.png'.format(
                    lensresolution, resolution, i))
            )
        plt.close()


newshape = 128
field = np.zeros(detectorspace.shape)
field[mask] = interpolator((isource)).val
from source_fwd import save_fits
imagepixels = get_pixel(detectorspace, imag)
for ii, pix in enumerate(imagepixels):
    masknew = np.zeros_like(mask)
    masknew[pix[1]-newshape//2:pix[1]+newshape//2,
            pix[0]-newshape//2:pix[0]+newshape//2] = True

    np.save(
        './output/fields/glamer_{}_field_{}_sr{}_lr{}'.format(
            lensresolution, ii, resolution, lensresolution),
        d[masknew].reshape((newshape,)*2)
    )
    np.save(
        './output/fields/interpolated_{}_field_{}_sr{}_lr{}'.format(
            lensresolution, ii, resolution, lensresolution),
        field[masknew].reshape((newshape,)*2)
    )
    save_fits(
        d[masknew].reshape((newshape,)*2),
        './output/fields/glamer_{}_field_{}_sr{}_lr{}'.format(
            lensresolution, ii, resolution, lensresolution)
    )
    save_fits(
        field[masknew].reshape((newshape,)*2),
        './output/fields/interpolated_{}_field_{}_sr{}_lr{}'.format(
            lensresolution, ii, resolution, lensresolution)
    )

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(d[masknew].reshape((newshape,)*2)+1, norm=LogNorm())
    axes[1].imshow(field[masknew].reshape((newshape,)*2)+1, norm=LogNorm())
    axes[2].imshow(
        d[masknew].reshape((newshape,)*2)+1-field[masknew].reshape((newshape,)*2)+1,
        norm=LogNorm())
    plt.show()

exit()

data = ift.makeField(ift.UnstructuredDomain(mask.sum()), d[mask].reshape(-1))
# data = interpolator_glamer(isource)
# data = interpolator(Trans(isource))
data = ift.makeField(
    data.domain,
    data.val + np.random.normal(scale=noise_scale, size=data.shape)
)


def power_spectrum(k):
    return 1000/(0.01+k)**3


imargs = {'extent': detectorspace.extent, 'origin': 'lower'}

harmonic_space = isspace.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=isspace)
power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = interpolator @ Trans @ HT

data_space = R.target
N = ift.ScalingOperator(data_space, noise_scale)
D_inv = R.adjoint @ N.inverse @ R + S.inverse
j = R.adjoint_times(N.inverse_times(data))
IC = ift.GradientNormController(iteration_limit=10000, tol_abs_gradnorm=1e-3)
print("Wiener")
D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
m = D(j)

field = np.zeros(detectorspace.shape)
field[mask] = interpolator(Trans(HT(m))).val
dd = np.zeros(detectorspace.shape)
dd[mask] = data.val

field[field<0] = 0.000000001

fig, axes = plt.subplots(6, 3, figsize=(20, 30))
ims = np.zeros_like(axes)
imagepixels = get_pixel(detectorspace, imag)
for ii, pix in enumerate(imagepixels):
    masknew = np.zeros_like(mask)
    masknew[pix[1]-50:pix[1]+50, pix[0]-50:pix[0]+50] = True
    axes[ii, 0].set_title('data')
    axes[ii, 1].set_title('Ls')
    axes[ii, 2].set_title('data - Ls')
    ims[ii, 0] = axes[ii, 0].imshow(dd[masknew].reshape(100, 100)+1, norm=LogNorm())
    ims[ii, 1] = axes[ii, 1].imshow(field[masknew].reshape(100, 100)+1, norm=LogNorm())
    ims[ii, 2] = axes[ii, 2].imshow(dd[masknew].reshape(100, 100)+1-field[masknew].reshape(100, 100)+1, cmap='RdBu_r')
ims[-1, 0] = axes[-1, 0].imshow(isource.val.T/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.0))
ims[-1, 1] = axes[-1, 1].imshow(HT(m).val/isource.val.max(), origin='lower', vmin=0.0001, vmax=1.0) # , norm=LogNorm(vmin=0.0001, vmax=1.0))
ims[-1, 2] = axes[-1, 2].imshow((isource.val.T-HT(m).val)/isource.val.max(), origin='lower', cmap='RdBu_r')
axes[-1, 0].set_title('source')
axes[-1, 1].set_title('rec')
axes[-1, 2].set_title('source - rec')
for im, ax in zip(ims.flatten(), axes.flatten()):
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(
    join('/home/jruestig/pro/python/lensing/output/interbeginning',
         'dglamer_glamerdeflection_WienerFilter_lr{}_sr{}_tmp.pdf'.format(
             lensresolution, resolution))
)
plt.close()


exit()

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
ims = np.zeros_like(axes)
ims[0, 0] = axes[0, 0].imshow(dd, **imargs)
ims[0, 1] = axes[0, 1].imshow(field, **imargs)
ims[0, 2] = axes[0, 2].imshow(dd-field, **imargs, cmap='RdBu_r')
for ax in axes[0]:
    ax.contour(mask, **imargs, cmap='Greys')
ims[1, 0] = axes[1, 0].imshow(isource.val.T/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.0))
ims[1, 1] = axes[1, 1].imshow(HT(m).val/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.0))
ims[1, 2] = axes[1, 2].imshow((isource.val.T-HT(m).val)/isource.val.max(), origin='lower', cmap='RdBu_r')
axes[0, 0].set_title('data')
axes[0, 1].set_title('Ls')
axes[0, 2].set_title('data - Ls')
axes[1, 0].set_title('source')
axes[1, 1].set_title('rec')
axes[1, 2].set_title('source - rec')
for im, ax in zip(ims.flatten(), axes.flatten()):
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(
    join('/home/jruestig/pro/python/lensing/output/interbeginning',
         'dglamer_modeldeflection_WienerFilter_lr{}_sr{}.pdf'.format(
             lensresolution, resolution))
)
plt.close()

exit()

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

N = ift.ScalingOperator(data.domain, 1./noise_scale, sampling_dtype=float)
likelihood_energy = (ift.GaussianEnergy(data=data, inverse_covariance=N) @ interpolator @ Trans @ correlated_field)
H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

initial_mean = ift.from_random(H.domain, 'normal')
mean = initial_mean

N_samples = 5
# Draw new samples to approximate the KL six times
for i in range(10):
    print(i, 'Samples:{}'.format(N_samples))

    if i > 1:
        ic_newton = ift.AbsDeltaEnergyController(name='Newton {}'.format(i), deltaE=0.05, convergence_level=1, iteration_limit=20)
    elif i > 2:
        print('Here')
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
    sfield = recsource
    # sfield = np.zeros_like(recsource)
    # maxvalue = isource.val.max()
    # sfield[recsource < maxvalue*2.0] = recsource[recsource < maxvalue*2.0]

    field = np.zeros(detectorspace.shape)
    field[mask] = interpolator(Trans(correlated_field(mean))).val

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(dd, **imargs, vmin=-0.10, vmax=dd.max()*0.01)
    ims[0, 1] = axes[0, 1].imshow(field, **imargs, vmin=-0.10, vmax=dd.max()*0.01)
    ims[0, 2] = axes[0, 2].imshow((dd-field)/noise_scale, **imargs, cmap='RdBu_r', vmin=-5., vmax=5.)
    ims[1, 0] = axes[1, 0].imshow(isource.val.T/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.5))
    ims[1, 1] = axes[1, 1].imshow(sfield/isource.val.max(), origin='lower', norm=LogNorm(vmin=0.0001, vmax=1.5))
    ims[1, 2] = axes[1, 2].imshow((isource.val.T-sfield)/isource.val.max(), origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[0, 0].set_title('data')
    axes[0, 1].set_title('Ls')
    axes[0, 2].set_title('(data - Ls)/noise')
    axes[1, 0].set_title('source')
    axes[1, 1].set_title('rec')
    axes[1, 2].set_title('source - rec')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(
        join('/home/jruestig/pro/python/lensing/output/interbeginning/KLs',
             'newmask_dglamer_defmodel_lr{}_sr{}_iter{}.png'.format(
                 lensresolution, resolution, i))
        )
    plt.close()
