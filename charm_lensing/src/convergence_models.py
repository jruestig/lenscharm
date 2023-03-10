#!/usr/bin/env python3
from functools import reduce, partial

import jax.numpy as jnp
import nifty8 as ift
from cluster_fits import Space
from jax import custom_jvp

from charm_lensing.src import utils
from charm_lensing.src import prior_handler

from sys import exit


def rotation(grid, theta):
    """
    Rotates the passed coordinates anticlockwise by an angle theta
    """
    x = grid[0] * jnp.cos(theta) - grid[1] * jnp.sin(theta)
    y = grid[0] * jnp.sin(theta) + grid[1] * jnp.cos(theta)
    return x, y


@custom_jvp
def F(x):
    def bigger(x):
        return (x**2-1)**(-1/2) * jnp.arctan(jnp.sqrt(x**2-1))

    def smaller(x):
        return (1-x**2)**(-1/2) * jnp.arctanh(jnp.sqrt(1-x**2))

    x0 = jnp.where(x > 1, bigger(x), 0)
    x1 = jnp.where(x < 1, smaller(x), 0)
    x2 = jnp.where(x == 1, 1, 0)

    return x0 + x1 + x2


@F.defjvp
def F_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = F(x)
    tangent_out = jnp.where(x != 1, (1 - x**2 * F(x)) / (x*(x**2 - 1)), 0)
    return primal_out, tangent_out * x_dot


def nfw_convergence(params, coords): #
    # convergence field (Keeton 2002 eq. 55)
    b, rs, center, theta, q = params

    x, y = rotation((coords[0]-center[0], coords[1]-center[1]), theta)
    R = jnp.hypot(q*x, y)

    x = R / rs
    return jnp.log(2*b*(1-F(x))/(x**2-1))


def piemd_convergence(params, coords):
    b, rs, center, theta, q = params

    f, bc = 1/q, rs/q
    x, y = rotation((coords[0]-center[0], coords[1]-center[1]), theta)
    return jnp.log(b*jnp.sqrt(f)/(2*jnp.sqrt(x**2+f**2*y**2+bc**2)))


def get_operator(ift_lens_space, prefix, model_cfg, model_type):
    coords = Space(ift_lens_space.shape, ift_lens_space.distances).xycoords

    if model_type == 'nfw':
        prior_keys = ['b', 'rs', 'center', 'theta', 'q']
        function = nfw_convergence
    elif model_type == 'piemd':
        prior_keys = ['b', 'rs', 'center', 'theta', 'q']
        function = piemd_convergence
    else:
        raise ValueError(f"Invalid model type '{model_type}', must be 'nfw' or 'piemd'")

    # Check that model_cfg contains all prior keys
    if not all(key in model_cfg for key in prior_keys):
        raise ValueError(f"The lens model does not contain all required the prior keys: {prior_keys}")

    model_keys = ['_'.join((prefix, key)) for key in prior_keys]

    model_priors = prior_handler.ParametricPriorHandler(prefix, model_cfg)
    free_parameters = model_priors.free_parameter_operator
    constant_parameters = model_priors.constant_parameter_operator

    MODEL = ift.JaxOperator(
        free_parameters.target,
        ift_lens_space,
        lambda x: function(
            (constant_parameters(x)[key] for key in model_keys),
            coords
        ))

    return {'mean_convergence': MODEL @ free_parameters,
            'mean_convergence_prior': free_parameters,
            'mean_convergence_constants': constant_parameters}


def get_convergence_model(cfg, ift_lens_space):
    cfm_maker = ift.CorrelatedFieldMaker('lens_')
    cfm_maker.set_amplitude_total_offset(**cfg['priors']['lens']['offset'])
    cfm_maker.add_fluctuations(ift_lens_space, **cfg['priors']['lens']['fluctuations'])
    perturbations_convergence = cfm_maker.finalize()
    perturbations_pspec = cfm_maker.power_spectrum

    # FIXME: Takes only one profile
    for key in cfg['priors']['lens']:
        if key.split('_')[0].lower() in ['nfw']:
            res = get_operator(ift_lens_space, key, cfg['priors']['lens'][key], 'nfw')

        elif key.split('_')[0].lower() in ['piemd']:
            res = get_operator(ift_lens_space, key, cfg['priors']['lens'][key], 'piemd')

    res['full_model_convergence'] = (res['mean_convergence'] + perturbations_convergence).exp()
    res['perturbations_convergence'] = perturbations_convergence
    res['perturbations_pspec'] = perturbations_pspec

    return res


if __name__ == '__main__':
    import yaml

    cfg_file = '/home/jruestig/pro/python/lensing/configs/first_config.yaml'
    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']
    ift_lens_space = ift.RGSpace(npix_lens, dist_lens)
    mean_model = get_operator(ift_lens_space, 'piemd', cfg['priors']['lens']['piemd'], 'piemd')

    exit()
