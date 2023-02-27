import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = jnp.linspace(0, 10.1, 100)


def model(pi):
    a, b = pi
    return a*(x - b)**2


pi_true = jnp.array((0.2, 3.1))
sigma = 0.1

d = model(pi_true) + np.random.normal(scale=sigma, size=x.shape)


def jointprob(pi):
    prediction = model(pi)
    return jnp.mean((d - prediction)**2)




@jax.jit
def update(pi, lr=1e-3):
    return pi - lr * jax.grad(jointprob)(pi)


pi = jnp.array([1., 1.])

pprev = 0.0
while np.sum(np.abs((pi - pprev))) > 1e-12:
    pprev = pi
    pi = update(pi)

plt.scatter(x, d)
plt.plot(x, model(pi))
print(pi_true, pi, pi_true-pi)
plt.show()



# Some lessons:
#
# 1) The learning rate is extremly important
# 2) The prior doesn't necesserily help
# 3) jax.grad by default differentiates the first argument, can be changed by argnums.
