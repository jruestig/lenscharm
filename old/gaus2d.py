import cluster_fits as cf
import matplotlib.pyplot as plt

import nifty8.re as jft

import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal   # .pdf(xy, 0, distance*kern)/4
from jax import jit
from jax import random
seed = 42
key = random.PRNGKey(seed)

from source_fwd import load_fits
from functools import partial
import numpy as np


psf = load_fits('/home/jruestig/pro/python/source_fwd/fits/psfs/psf_slacs.fits')

xspace = cf.Space(psf.shape, 0.13)



def source(primals):
    # A, x0, y0, a00, a01, a11 = primals

    x = (xspace.xycoords[0] - primals[1])
    y = (xspace.xycoords[1] - primals[2])

    return primals[0]*jnp.exp(-0.5*(
        x*primals[3]*x + x*primals[4]*y + y*primals[4]*x + y*primals[5]*y
    ))



def Gaussian(data, noise_cov_inv_sqrt):

    def hamiltonian(primals):
        p_res = source(primals) - data
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

n_scale = 0.001
data = psf + np.random.normal(scale=n_scale, size=psf.shape)
noise_cov_inv_sqrt = lambda x: 1/np.sqrt(n_scale)

signal_response = source
nll = Gaussian(data, noise_cov_inv_sqrt) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

key, subkey = random.split(key)
pos_init = jft.Field(jnp.array([1., 0, 0, 1, 1, 0]))
pos = jft.Field(pos_init.val)

n_newton_iterations = 10
# Maximize the posterior using natural gradient scaling
pos = jft.newton_cg(
    fun_and_grad=ham_vg, x0=pos.val, hessp=ham_metric, maxiter=n_newton_iterations, energy_reduction_factor=0.001,
)


fig, ax = plt.subplots()
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
# ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, '.', alpha=0.7, label="Data")
ax.plot(signal_response(pos), alpha=0.7, label="Reconstruction")
ax.legend()
fig.tight_layout()
plt.show()
