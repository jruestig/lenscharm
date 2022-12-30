#!/usr/bin/env python3

import nifty8 as ift
import cluster_fits as cf
import matplotlib.pyplot as plt
import numpy as np

from os.path import join, exists
from NiftyOperators import PriorTransform
from jax import numpy as jnp

from sys import exit

xshape, xfov = 128, 60
xspace = cf.Space((xshape,)*2, xfov/xshape)

lens = cf.SingularPowerLaw(xspace)
source = cf.Sersic(xspace)


def Model(target, lens, source):
    def brightness(x):
        xy, xs = x
        return source.brightness(
            xy, xs
        ).sum(axis=-1).reshape(target.shape)

    def model(x):
        xs = source.get_params(x)
        xl = lens.get_params(x)

        xy = target.xycoords - lens.deflection(
            target.xycoords, xl
        ).sum(axis=0).reshape(target.shape)
        # xy = target.xycoords

        return brightness((xy, xs))
    return model


xtrue = {'SPWL_0_b': 10.2,
         'SPWL_0_t': 2.0,
         'SPWL_0_x0': 0.0,
         'SPWL_0_y0': 0.0,
         'SPWL_0_th': 0.0,
         'SPWL_0_q': 1.0,
         'Sersic_0_Ie': 2.0,
         'Sersic_0_Re': 13.0,
         'Sersic_0_n': 2.0,
         'Sersic_0_x0': 0.1,
         'Sersic_0_y0': 0.2,
         'Sersic_0_q': 1.0,
         'Sersic_0_th': 0.0}

model = Model(xspace, lens, source)
prior = PriorTransform((lens+source).get_priorparams())

data = model(xtrue) + np.random.normal(size=xspace.shape)


cdata = ift.makeField(
    ift.RGSpace(xspace.shape, xspace.distances),
    data)
imodel = ift.JaxOperator(prior.domain, cdata.domain, model)
eg = ift.GaussianEnergy(cdata) @ imodel @ prior


# Minimization parameters
ic_sampling = ift.AbsDeltaEnergyController(
    name="Sampling (linear)", deltaE=0.05, iteration_limit=100)
ic_newton = ift.AbsDeltaEnergyController(
    name='Newton', deltaE=0.5, convergence_level=2, iteration_limit=30)
ic_sampling_nl = ift.AbsDeltaEnergyController(
    name='Sampling (nonlin)', deltaE=0.5, iteration_limit=15, convergence_level=2)

minimizer = ift.NewtonCG(ic_newton)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

# Minimize KL
n_samples = 20

position = ift.full(prior.domain, 0.)

samples, position = ift.optimize_kl(
    eg, 1, n_samples,
    minimizer, ic_sampling, minimizer_sampling,
    initial_position=position,
    # export_operator_outputs={"signal": prior},
    output_directory=join('output', 'sersic_sis'),
    return_final_position=True,
)
for ii in range(10): plt.close()
mean, var = samples.sample_stat(prior)
samples_prior = (samples, prior)

mean = prior(position)

fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(data)
axes[0, 1].imshow(model(mean.val))
axes[0, 2].imshow(data-model(mean.val))
axes[1, 0].imshow(source.brightness_field(xtrue))
axes[1, 1].imshow(source.brightness_field(mean.val))
axes[1, 2].imshow(source.brightness_field(xtrue)-source.brightness_field(mean.val))
plt.show()
