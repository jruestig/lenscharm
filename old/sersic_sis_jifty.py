import cluster_fits as cf
import matplotlib.pyplot as plt

import nifty8.re as jft
from jax import jit
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)

import sys

from functools import partial

from jax import numpy as jnp


xshape, xfov = 128, 60
xspace = cf.Space((xshape,)*2, xfov/xshape)

lens = cf.SingularPowerLaw(xspace)
source = cf.Sersic(xspace)


xtrue = {'SPWL_0_b': 1.2,
         'SPWL_0_t': 1.0,
         'SPWL_0_x0': 0.0,
         'SPWL_0_y0': 0.0,
         'SPWL_0_th': 0.0,
         'SPWL_0_q': 0.0,
         'Sersic_0_Ie': 2.0,
         'Sersic_0_Re': 3.0,
         'Sersic_0_n': 2.0,
         'Sersic_0_x0': 0.1,
         'Sersic_0_y0': 0.2,
         'Sersic_0_q': 0.0,
         'Sersic_0_th': 0.0}


def Model(target, lens, source):
    lenslength = len(lens.domain)
    sourcelength = len(source.domain)

    lognorm = jft.lognormal_prior(4, 3)
    uniform = jft.uniform_prior(0, 2)
    uniformq = jft.uniform_prior(1, 10)
    gauss = jft.normal_prior(0, 10)

    def lens_prior(x):
        b, t, x0, y0, th, q = x
        b = lognorm(b)
        t = uniform(t)
        x0 = gauss(x0)
        y0 = gauss(y0)
        th = gauss(th)
        q = uniformq(q)
        return jnp.array((b, t, x0, y0, th, q))

    def source_prior(x):
        ie, re, n, x0, y0, q, th = x
        ie = lognorm(ie)
        re = lognorm(re)
        n = uniform(n)
        x0 = gauss(x0)
        y0 = gauss(y0)
        th = gauss(th)
        q = uniformq(q)
        return jnp.array((ie, re, n, x0, y0, q, th))

    def brightness(x):
        xy, xs = x
        return source.brightness(
            xy, xs
        ).sum(axis=-1).reshape(target.shape)

    def model(x):
        xl = lens_prior(x[:lenslength])
        xs = source_prior(x[lenslength:])

        xy = target.xycoords - lens.deflection(
            target.xycoords, xl
        ).sum(axis=0).reshape(target.shape)
        # xy = target.xycoords

        return brightness((xy, xs))
    return model


model = Model(xspace, lens, source)

def toarray(x):
    return jnp.array([xi for xi in x.values()])


def Gaussian(data, noise_cov_inv_sqrt):

    def hamiltonian(primals):
        p_res = primals - data
        l_res = noise_cov_inv_sqrt(p_res)
        return 0.5 * jnp.sum(l_res**2)

    def left_sqrt_metric(primals, tangents):
        return noise_cov_inv_sqrt(tangents)

    lsm_tangents_shape = jnp.shape(data)
    # Better: `tree_map(ShapeWithDtype.from_leave, data)`

    return jft.Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


seed = 42
key = random.PRNGKey(seed)
dims = xspace.shape
pos_truth = jft.Field(toarray(xtrue))

noise_cov_inv_sqrt = lambda x: 1.5**-1 * x

signal_response = model
signal_response_truth = signal_response(pos_truth)
noise_truth = 1. / noise_cov_inv_sqrt(jnp.ones(dims)) * random.normal(shape=dims, key=key)
data = signal_response_truth + noise_truth

nll = Gaussian(data, noise_cov_inv_sqrt) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

key, subkey = random.split(key)
pos_init = jft.Field(jnp.ones_like(pos_truth.val))
pos = jft.Field(pos_init.val)

n_newton_iterations = 10
# Maximize the posterior using natural gradient scaling
pos = jft.newton_cg(
    fun_and_grad=ham_vg, x0=pos, hessp=ham_metric, maxiter=n_newton_iterations
)

fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(data)
b.imshow(signal_response(pos))
c.imshow(data-signal_response(pos))
plt.show()


n_mgvi_iterations = 5
n_samples = 10
n_newton_iterations = 20

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
    )

    print("Minimizing...", file=sys.stderr)
    pos = jft.newton_cg(
        fun_and_grad=partial(ham_vg, primals_samples=mg_samples),
        x0=pos,
        hessp=partial(ham_metric, primals_samples=mg_samples),
        maxiter=n_newton_iterations
    )
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)


fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(data)
b.imshow(signal_response(pos))
c.imshow(data-signal_response(pos))
plt.show()
