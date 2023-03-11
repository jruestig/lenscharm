#!/usr/bin/env python3
import jax.numpy as jnp
from charm_lensing.src import prior_handler
import nifty8 as ift
from cluster_fits import Space


def shear_deflection(params, coords):
    ss, sa, center = params
    xx, yy = coords - center.reshape(2, 1, -1)
    ax = jnp.cos(2*sa)*xx + jnp.sin(2*sa)*yy
    ay = jnp.sin(2*sa)*xx - jnp.cos(2*sa)*yy
    return ss*jnp.array((ax, ay))


def get_shear_operator(ift_lens_space, prefix, shear_cfg, points_domain):
    coords = Space(ift_lens_space.shape, ift_lens_space.distances).xycoords

    prior_keys = ['strength', 'theta', 'center']
    model_keys = ['_'.join((prefix, key)) for key in prior_keys]

    priors = prior_handler.ParamatricPrior(prefix, shear_cfg)
    free_parameters = priors.free_parameter_operator
    constant_parameters = priors.constant_parameter_operator

    deflection_shear = ift.JaxOperator(
        free_parameters.target,
        points_domain,
        lambda x: shear_deflection(
            (constant_parameters(x)[key] for key in model_keys),
            coords
        ).reshape(2, -1)
    )

    return {
        'deflection_shear': deflection_shear @ free_parameters,
        'deflection_shear_prior': free_parameters,
        'deflection_shear_constants': constant_parameters
    }


def get_shear_model(cfg, points_domain, ift_lens_space):
    shear_dict = None
    for key in cfg['priors']['lens']:
        if key.split('_')[0].lower() in ['shear']:
            shear_dict = get_shear_operator(
                ift_lens_space, key, cfg['priors']['lens'][key],
                points_domain
            )

    return shear_dict


if __name__ == '__main__':
    pass
