#!/usr/bin/env python3
import jax.numpy as jnp
import nifty8 as ift
from nifty8 import JaxOperator
from cluster_fits import Space, EllipticalNfw
from jax import custom_jvp
from functools import reduce


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


def rotation(grid, theta):
    """
    Rotates the passed coordinates anticlockwise by an angle theta
    """
    x = grid[0] * jnp.cos(theta) - grid[1] * jnp.sin(theta)
    y = grid[0] * jnp.sin(theta) + grid[1] * jnp.cos(theta)
    return x, y


def nfw_convergence(params, coords): #
    # convergence field (Keeton 2002 eq. 55)
    b, rs, center, theta, q = params

    x, y = rotation((coords[0]-center[0], coords[1]-center[1]), theta)
    R = jnp.hypot(q*x, y)

    x = R / rs
    return jnp.log(2*b*(1-F(x))/(x**2-1))


def distribution_getter(prefix, values):
    distribution = values.pop('distribution')
    values['key'] = prefix + values['key']

    if distribution in ['uniform']:
        return ift.UniformOperator(
            ift.UnstructuredDomain(values['N_copies']),
            loc=values['mean'],
            scale=values['sigma']
        ).ducktape(values['key']).ducktape_left(values['key'])

    if distribution in ['normal']:
        return ift.NormalTransform(**values).ducktape_left(values['key'])

    if distribution in ['log_normal', 'lognormal']:
        return ift.LognormalTransform(**values).ducktape_left(values['key'])

    # FIXME: Something which justs inserts a value as an operator,
    # i.e. conditional to that value
    if distribution is None:
        value = ift.Field.from_raw(
            ift.UnstructuredDomain(values['N_copies']), values['mean']
        )
        return ift.DiagonalOperator(value).ducktape_left(values['key']).ducktape(values['key'])


def get_nfw_operator(ift_lens_space, cfg):
    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']
    coords = Space(npix_lens, dist_lens).xycoords

    prefix = cfg['priors']['lens']['nfw']['prefix']
    prior_keys = ['b', 'rs', 'center', 'theta', 'q']
    prior_transform = []
    for key in prior_keys:
        prior_transform.append(
            distribution_getter(
                prefix,
                cfg['priors']['lens']['nfw'][key],
            )
        )
    prior_transform = reduce(lambda x,y: x+y, prior_transform)

    prior_keys = [prefix + key for key in prior_keys]
    ENFW = ift.JaxOperator(
        prior_transform.target,
        ift_lens_space,
        lambda x: nfw_convergence(
            (x[key] for key in prior_keys),
            coords
        ))

    return {'convergence_mean': ENFW @ prior_transform,
            'prior_transform': prior_transform}


def get_convergence_model(cfg):
    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']
    ift_lens_space = ift.RGSpace(npix_lens, dist_lens)

    cfm_maker = ift.CorrelatedFieldMaker('lens_')
    cfm_maker.set_amplitude_total_offset(**cfg['priors']['lens']['offset'])
    cfm_maker.add_fluctuations(ift_lens_space, **cfg['priors']['lens']['fluctuations'])
    correlated_convergence = cfm_maker.finalize()

    correlated_pspec = cfm_maker.power_spectrum

    res = get_nfw_operator(ift_lens_space, cfg)

    res['convergence'] = (res['convergence_mean'] + correlated_convergence).exp()
    res['correlated_convergence'] = correlated_convergence
    res['correlated_pspec'] = correlated_pspec

    return res


if __name__ == '__main__':
    from sys import exit
    import yaml

    cfg_file = '/home/jruestig/pro/python/lensing/configs/first_config.yaml'
    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    npix_lens = cfg['spaces']['lens_space']['Npix']
    dist_lens = cfg['spaces']['lens_space']['distance']
    ift_lens_space = ift.RGSpace(npix_lens, dist_lens)
    ENFW = get_nfw_operator(ift_lens_space, cfg)
