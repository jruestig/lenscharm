#!/usr/bin/env python3

from jax import jit
from jax import random
from jax import numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial

import sys
from sys import exit

import matplotlib.pyplot as plt

import nifty8.re as jft


x = jnp.arange(0, 10, 0.1)

def model(primals):
    a, b, c = primals
    val = a*(x + b**2)
    return x ** c


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
dims = x.shape
pos_truth = jft.Field(jnp.array((10., 2., 3.)))


noise_cov_inv_sqrt = lambda x: 10.1**-1 * x

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
    fun_and_grad=ham_vg, x0=pos, hessp=ham_metric, maxiter=n_newton_iterations, energy_reduction_factor=0.001,
)


fig, ax = plt.subplots()
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
# ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, '.', alpha=0.7, label="Data")
ax.plot(signal_response(pos), alpha=0.7, label="Reconstruction")
ax.legend()
fig.tight_layout()
plt.show()


n_mgvi_iterations = 3
n_samples = 10
n_newton_iterations = 50

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=False,
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


fig, (ax, ay) = plt.subplots(2, 1)
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
# ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, '.', alpha=0.7, label="Data")
ax.plot(signal_response(pos), alpha=0.7, label="Reconstruction")
ax.legend()

ay.plot(signal_response_truth-signal_response(pos), label='Residuals')
ay.plot(noise_truth, '.', alpha=0.7, label="Noise")
ay.legend()
fig.tight_layout()
plt.show()



# Lessons:
# when formalizing a model, jax takes it easier when
# val = a * (x - b)
# val = val**c
#
# instead of
# val = (a * (x - b)) ** c
