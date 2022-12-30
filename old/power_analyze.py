import cluster_fits as cf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import nifty8 as ift

from os.path import join
from sys import exit

import jax.numpy as jnp
import jax
from scipy.optimize import minimize


space = cf.Space((1024,)*2, 1.)
klcoords = space.klcoords
box = cf.make_box(space.shape, (128,)*2, space.distances[0])


powers =[]
for ii in range(100):
    field = np.array(h5py.File(join('/home/jruestig/Data/angulo/tcm_clusters/', 'big_clusters_adj_{}.h5'.format(ii)))['data'])

    ispace = ift.RGSpace((128,)*2, 1.)
    f = ift.makeField(ispace, field[box['slice']])
    FFT = ift.FFTOperator(ispace.get_default_codomain(), ispace)
    ff = FFT.adjoint(f)
    powers.append(ift.power_analyze(ff).val)


d = jnp.array([np.log(p) for p in powers])
x = np.log(np.arange(powers[0].shape[0])+1)


def f1(a, b, x):
    return a+b*x

def f2(a, b, c, x):
    return a+b*x+c*x**2

def f3(a, b, c, d, x):
    return a+b*x+c*x**2+d*x**3

def f4(a, b, c, d, e, x):
    return a+b*x+c*x**2+d*x**3+e*x**4

def fun(f, p):
    return jnp.sum((f(*p, x) - d)**2)


fun1 = jax.jit(lambda p: fun(f1, p))
fun2 = jax.jit(lambda p: fun(f2, p))
fun3 = jax.jit(lambda p: fun(f3, p))
fun4 = jax.jit(lambda p: fun(f4, p))


sols = []
for f, start in zip((fun1, fun2, fun3, fun4), (np.ones(2), np.ones(3), np.ones(4), np.ones(5))):
    sols.append(
        minimize(f, start, jac=jax.grad(f), method='BFGS')
    )


plt.figure(figsize=(18, 12))
xn = np.log(np.arange(powers[0].shape[0])+1)
for p in powers:
    plt.plot(np.arange(powers[0].shape[0]), p, '.')

plt.plot(np.arange(powers[0].shape[0]), np.exp(f1(*sols[0]['x'], xn)), label='f1')
plt.plot(np.arange(powers[0].shape[0]), np.exp(f2(*sols[1]['x'], xn)), label='f2')
plt.plot(np.arange(powers[0].shape[0]), np.exp(f3(*sols[2]['x'], xn)), label='f3')
plt.plot(np.arange(powers[0].shape[0]), np.exp(f4(*sols[3]['x'], xn)), label='f4')
plt.legend()
plt.loglog()
plt.tight_layout()
plt.show()
#plt.savefig('plots/least_squares_ps')
