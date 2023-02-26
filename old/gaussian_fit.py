import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = jnp.linspace(-10, 10.1, 100)
x = np.meshgrid(x, x)
y = x[0]**2 + x[1]**2

pi_true = jnp.array((0.2, 1.0, 3.3, 1.2, 3.2))


def f(d, e):
    return (x[0]-d)**2 + (x[1]-e)**2

def modelnew(pi):
    a, b, c, d, e = pi
    return a * jnp.exp(-0.5*b*(f(d, e)-c)**2)

def loss(pi, d):
    return jnp.mean((d - modelnew(pi))**2)

@jax.jit
def update(pi, d, lr=1e-3):
    return pi - lr * jax.grad(loss)(pi, d)


pi = jnp.array((1., 1., 1., 1., 1.))
dnew = modelnew(pi_true) + np.random.normal(scale=0.001, size=y.shape)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(dnew)
axes[1].imshow(modelnew(pi))
axes[2].imshow(dnew-modelnew(pi))
plt.show()

for _ in range(50000):
    pi = update(pi, dnew, 2.15)
    print(pi)


fig, axes = plt.subplots(1, 3)
axes[0].imshow(dnew)
axes[1].imshow(modelnew(pi))
axes[2].imshow(dnew-modelnew(pi))
plt.show()
