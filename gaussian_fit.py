import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = jnp.linspace(0, 10.1, 100)

pi_true = jnp.array((0.2, 1.0, 0.3))



def modelnew(pi):
    a, b, c = pi
    return c * jnp.exp(-0.5*a*(x-b)**2)

def loss(pi, d):
    return jnp.mean((d - modelnew(pi))**2)

@jax.jit
def update(pi, d, lr=1e-3):
    return pi - lr * jax.grad(loss)(pi, d)

pi = jnp.array((1., 1., 1.))
dnew = modelnew(pi_true)

plt.scatter(x, dnew)
plt.plot(x, modelnew(pi))
plt.show()


for _ in range(2000):
    pi = update(pi, dnew, 1.)
    print(pi)



plt.scatter(x, dnew)
plt.plot(x, modelnew(pi))
plt.show()
