import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = jnp.linspace(-10, 10.1, 100)
x = np.meshgrid(x, x)
x = x[0]**2 + x[1]**2

pi_true = jnp.array((0.2, 1.0, 3.3))


# def f(c):
#     a, b = c
#     return x

def modelnew(pi):
    a, b, c = pi
    return a * jnp.exp(-0.5*b*(x-c)**2)

def loss(pi, d):
    return jnp.mean((d - modelnew(pi))**2)

@jax.jit
def update(pi, d, lr=1e-3):
    return pi - lr * jax.grad(loss)(pi, d)


pi = jnp.array((1., 1., 1.))
dnew = modelnew(pi_true) + np.random.normal(scale=0.001, size=x.shape)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(dnew)
axes[1].imshow(modelnew(pi))
axes[2].imshow(dnew-modelnew(pi))
plt.show()

pi = jnp.array((1., 1., 1.))
for _ in range(40000):
    pi = update(pi, dnew, 1.15)
    print(pi)


fig, axes = plt.subplots(1, 3)
axes[0].imshow(dnew)
axes[1].imshow(modelnew(pi))
axes[2].imshow(dnew-modelnew(pi))
plt.show()
