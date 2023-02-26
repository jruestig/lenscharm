#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline


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
    deflectionf = deflection(convergence_model.force(mean)).val.reshape(
        2, *convergence.shape
    )

    # number_of_levels = 5
    # levels = np.linspace(start=0.1, stop=1, num=number_of_levels)*cdata.max()

    vmax = np.hypot(*deflection_data).max()
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # imc = axes[0, 0].contour(convergence_data, colors='red', levels=levels, origin='lower', extent=extent)
    # axes[0, 0].contour(convergence, colors='orange', levels=levels, origin='lower', extent=extent)
    # axes[0, 1].contour(convergence_data, colors='red', levels=levels, origin='lower', extent=extent)
    # axes[0, 1].contour(convergence, colors='orange', levels=levels, origin='lower', extent=extent)

    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        convergence_data,
        origin='lower',
        vmin=convergence_data.min(), vmax=convergence_data.max(),
        cmap='RdYlBu_r',
        extent=extent)  # norm=LogNorm(vmax=convergence_data.max(),vmin=convergence_data.min()))
    ims[0, 1] = axes[0, 1].imshow(
        convergence,
        origin='lower',
        vmax=convergence_data.max(), vmin=convergence_data.min(),
        cmap='RdYlBu_r',
        extent=extent)  # norm=LogNorm(vmax=convergence_data.max(),vmin=convergence_data.min()))
    ims[0, 2] = axes[0, 2].imshow(
        (convergence_data-convergence)/convergence_data.max(), origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3, extent=extent)
    ims[1, 0] = axes[1, 0].imshow(
        np.hypot(*deflection_data), vmin=-0.10, origin='lower', vmax=vmax, extent=extent)
    ims[1, 1] = axes[1, 1].imshow(
        np.hypot(*deflectionf), vmin=-0.10, origin='lower', vmax=vmax, extent=extent)
    ims[1, 2] = axes[1, 2].imshow(
        np.hypot(*(deflection_data-deflectionf))/vmax,
        vmin=-0.3, vmax=0.3,
        origin='lower', cmap='RdBu_r', extent=extent)
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
        scale = true_source.max()
    else:
        scale = None

    if cast_to_size:
        xycoord = np.linspace(0, 1, num=source_reconstruction.shape[0])
        Recaster = RectBivariateSpline(
            xycoord, xycoord, source_reconstruction
        )
        xycoordnew = np.linspace(0, 1, num=true_source.shape[0])
        source_reconstruction = Recaster(*(xycoordnew,)*2, grid=True) * (
            source_reconstruction.shape[0]/true_source.shape[0]
        )**2 / 4 # FIXME: This 4 should not be there

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ims = np.zeros_like(axes)
    ims[0, 0] = axes[0, 0].imshow(
        true_source, origin='lower', vmin=0, vmax=scale, extent=extent)
    ims[0, 1] = axes[0, 1].imshow(
        source_reconstruction, origin='lower', vmin=0, vmax=scale,
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
