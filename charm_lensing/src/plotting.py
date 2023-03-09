#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
import nifty8 as ift


def deflection_check(samples_list,
                     ii,
                     outputdir=None,
                     convergence_model=None,
                     deflection=None,
                     deflection_data=None,
                     convergence_data=None,
                     extent=None):
    mean, var = samples_list.sample_stat()

    convergence = convergence_model.force(mean).val
    deflection_field = deflection(convergence_model.force(mean).val).reshape(
        2, *convergence.shape
    )

    # number_of_levels = 5
    # levels = np.linspace(start=0.1, stop=1, num=number_of_levels)*cdata.max()

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # imc = axes[0, 0].contour(convergence_data, colors='red', levels=levels, origin='lower', extent=extent)
    # axes[0, 0].contour(convergence, colors='orange', levels=levels, origin='lower', extent=extent)
    # axes[0, 1].contour(convergence_data, colors='red', levels=levels, origin='lower', extent=extent)
    # axes[0, 1].contour(convergence, colors='orange', levels=levels, origin='lower', extent=extent)

    if convergence_data is None:
        convergence_min = None
        convergence_max = None
        convergence_data = np.ones_like(convergence)
    else:
        convergence_min = convergence_data.min()
        convergence_max = convergence_data.max()

    if deflection_data is None:
        deflection_max = None
        deflection_data = np.zeros_like(deflection_field)
    else:
        deflection_max = np.hypot(*deflection_data).max()


    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        convergence_data,
        origin='lower',
        vmin=convergence_min,
        vmax=convergence_max,
        cmap='RdYlBu_r',
        extent=extent)
    # norm=LogNorm(vmax=convergence_data.max(),vmin=convergence_data.min()))
    ims[0, 1] = axes[0, 1].imshow(
        convergence,
        origin='lower',
        vmin=convergence_min,
        vmax=convergence_max,
        cmap='RdYlBu_r',
        extent=extent)  # norm=LogNorm(vmax=convergence_data.max(),vmin=convergence_data.min()))
    ims[0, 2] = axes[0, 2].imshow(
        (convergence_data-convergence)/convergence_data.max(),
        origin='lower',
        cmap='RdBu_r',
        vmin=-0.3,
        vmax=0.3,
        extent=extent)
    ims[1, 0] = axes[1, 0].imshow(
        np.hypot(*deflection_data),
        origin='lower',
        vmin=-0.10,
        vmax=deflection_max,
        extent=extent)
    ims[1, 1] = axes[1, 1].imshow(
        np.hypot(*deflection_field),
        origin='lower',
        vmin=-0.10,
        vmax=deflection_max,
        extent=extent)
    ims[1, 2] = axes[1, 2].imshow(
        np.hypot(*(deflection_data-deflection_field))/deflection_field.max(),
        vmin=-0.3,
        vmax=0.3,
        origin='lower',
        cmap='RdBu_r',
        extent=extent)
    axes[0, 0].set_title('convergence')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('(convergence - rec)/maxconvergence')
    axes[1, 0].set_title('deflection')
    axes[1, 1].set_title('reconstruction')
    axes[1, 2].set_title('(deflection - reconstruction)/maxdeflection')
    for kk, (im, ax) in enumerate(zip(ims.flatten(), axes.flatten())):
        cb = plt.colorbar(im, ax=ax)
        # if kk in [0, 1]:
        #     cb.add_lines(imc)

    plt.tight_layout()
    plt.savefig(f'{outputdir}/deflection_KL_{ii}.png')
    plt.close()


def Ls_check(
        samples_list,
        ii,
        outputdir=None,
        source_model=None,
        forward_model=None,
        true_source=None,
        data=None,
        noise_scale=None,
        extent=None,
        samescale=True,
        cast_to_size=True,
):
    mean, var = samples_list.sample_stat()

    source_reconstruction = source_model.force(mean).val.T
    source_std = source_model.force(var).sqrt().val.T
    dfield = forward_model(mean).val

    if samescale:
        source_max = true_source.max()
    else:
        source_max = None

    if cast_to_size:
        xycoord = np.linspace(0, 1, num=source_reconstruction.shape[0])
        Recaster = RectBivariateSpline(
            xycoord, xycoord, source_reconstruction
        )
        xycoordnew = np.linspace(0, 1, num=true_source.shape[0])
        source_reconstruction = Recaster(*(xycoordnew,)*2, grid=True) * (
            source_reconstruction.shape[0]/true_source.shape[0]
        )**2  # / 4 # FIXME: This 4 should not be there

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        true_source, origin='lower', vmin=0, vmax=source_max, extent=extent)
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower', vmin=0, vmax=source_max,
        extent=extent)
    ims[0, 2] = axes[0, 2].imshow(
        (true_source-source_reconstruction)/true_source.max(),
        origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3, extent=extent)
    ims[1, 0] = axes[1, 0].imshow(
        data, vmin=-0.10, origin='lower', vmax=data.max(), extent=extent)
    ims[1, 1] = axes[1, 1].imshow(
        dfield, vmin=-0.10, origin='lower', vmax=data.max(), extent=extent)
    ims[1, 2] = axes[1, 2].imshow(
        (data-dfield)/noise_scale,
        vmin=-3, vmax=3, origin='lower', cmap='RdBu_r', extent=extent)
    axes[0, 0].set_title('source')
    axes[0, 1].set_title('rec')
    axes[0, 2].set_title('(source - rec)/std')
    axes[1, 0].set_title('data')
    axes[1, 1].set_title('BLs')
    axes[1, 2].set_title('(data - BLs)/noisescale')
    for im, ax in zip(ims.flatten(), axes.flatten()):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if outputdir is None:
        plt.show()
    else:
        plt.savefig(f'{outputdir}/gauss_KL_{ii}.png')
    plt.close()


def prior_samples_plotting(full_model, convergence_dict, source_dict, real_source, data, extent):
    imargs = {'extent': extent, 'origin': 'lower'}

    prior_pos = ift.from_random(full_model.domain)

    vals = convergence_dict['mean_convergence_prior'].force(prior_pos).val
    for key, val in vals.items():
        print(key, val)

    Ls_model = full_model(prior_pos)
    full_source = source_dict['source_diffuse'].force(prior_pos)
    perturbations_source = source_dict['source_matern'].force(prior_pos).exp()

    full_convergence = convergence_dict['full_model_convergence'].force(prior_pos)
    mean_convergence = convergence_dict['mean_convergence'].force(prior_pos).exp()
    perturbations_convergence = convergence_dict['perturbations_convergence'].force(prior_pos).exp()

    fig, axes = plt.subplots(3, 3)
    # Source
    im = axes[0, 0].imshow(real_source, **imargs)
    plt.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title('real_source')

    im = axes[0, 1].imshow(full_source.val.T, **imargs)
    plt.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title('source_prior')

    im = axes[0, 2].imshow(perturbations_source.val.T, **imargs)
    plt.colorbar(im, ax=axes[0, 2])
    axes[0, 2].set_title('source_matern')

    # Ls
    im = axes[1, 0].imshow(data, **imargs)
    plt.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('data')

    im = axes[1, 1].imshow(Ls_model.val.T, **imargs)
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title('Ls model')

    # Mean
    im = axes[2, 0].imshow(mean_convergence.val.T, **imargs)
    plt.colorbar(im, ax=axes[2, 0])
    axes[2, 0].set_title(f'mean convergence')

    # Convergence Perturbations
    im = axes[2, 1].imshow(perturbations_convergence.val.T, **imargs)
    plt.colorbar(im, ax=axes[2, 1])
    axes[2, 1].set_title(f'Perturbations convergence')

    # Kappa
    im = axes[2, 2].imshow(full_convergence.val.T, **imargs)
    plt.colorbar(im, ax=axes[2, 2])
    axes[2, 2].set_title('full model convergence')

    print()
    plt.show()
    plt.close()
