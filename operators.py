import numpy as np
import jax.numpy as jnp
import nifty8 as ift


def jax_gaussian(domain):
    dist_x = domain.distances[0]
    dist_y = domain.distances[1]

    # Periodic Boundary conditions
    x_ax = np.arange(domain.shape[0]) * dist_x
    # x_ax = np.minimum(x_ax, domain.shape[0] - x_ax) * dist_x
    y_ax = np.arange(domain.shape[1]) * dist_x
    # y_ax = np.minimum(y_ax, domain.shape[1] - y_ax) * dist_y

    center = (domain.shape[0]//2,)*2
    x_ax -= center[0] * dist_x
    y_ax -= center[1] * dist_y
    X, Y = jnp.meshgrid(x_ax, y_ax, indexing='ij')

    def gaussian(var):
        return - (0.5 / var) * (X ** 2 + Y ** 2)

    # normalized psf
    return gaussian


class Reshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return ift.Field.from_raw(
                self._target, x.val.reshape(self._target.shape)
            )
        else:
            return ift.Field.from_raw(
                self._domain, x.val.reshape(self._domain.shape)
            )
