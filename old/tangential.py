import cluster_fits as cf
import numpy as np
import matplotlib.pyplot as plt
import jax

from scipy.stats import multivariate_normal   # .pdf(xy, 0, distance*kern)/4


def putit(val):
    return {'Sis_0_b': val,  'Sis_0_x0': 0., 'Sis_0_y0': 0.}


space = cf.Space((64,)*2, 0.1)
sis = cf.SisModel(space, xy0=np.array((0., 0.)))
pos = putit(1.)


sy = multivariate_normal.pdf(space.xycoords.T, mean=(0, 0), cov=1)
d = multivariate_normal.pdf((space.xycoords - sis.deflection_field(pos)).T, mean=(0, 0))


dsy = np.fft.ifft2(np.fft.fft2(sy)*2j*np.pi*space.klcoords).real
dsy0 = dsy[1]
dsy1 = dsy[0]


postest = cf.full(sis.domain, 1.2)
position = postest

th = np.arctan2(space.xycoords[1], space.xycoords[0])
durr = lambda position: -np.dot(
    (d - multivariate_normal.pdf((space.xycoords - sis.deflection_field(position)).T, mean=(0, 0))).reshape(-1),
    (dsy0*np.cos(th)+dsy1*np.sin(th)).reshape(-1))


params = (np.arange(3) + 1.2).reshape(1, 3)
# params = sis.get_params(position)
model = lambda p: sis.deflection(space.xycoords, p).sum(-1).reshape(2, -1)

residuum = ((d - multivariate_normal.pdf((space.xycoords - sis.deflection_field(position)).T, mean=(0, 0)))).reshape(-1)
dfdy = dsy.reshape(2, -1)
dadpi = lambda primals, tangents: jax.jvp(model, (primals,), (tangents,))[1]
dadpi = dadpi(params, params)

ein = np.einsum('ij, ij->j', dfdy, dadpi)

@jax.custom_jvp
def GaussSource(params):
    return multivariate_normal.pdf(
       (space.xycoords - sis.deflection(space.xycoords, params).sum(-1)).T.reshape(-1, 2),
        mean=(0, 0)
    )

@GaussSource.defjvp
def GaussSource_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = GaussSource(x)
    tangent_out = primal_out * jax.numpy.sum(
        (space.xycoords - sis.deflection(space.xycoords, x).sum(-1)) *\
        jax.jvp(lambda p: sis.deflection(space.xycoords, p).sum(-1), primals, tangents)[1],
        axis=0)
    return primal_out, tangent_out * x_dot
