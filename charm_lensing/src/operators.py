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

    # def gaussian(parameters):
    #     var, center = parameters
    #     return - (0.5 / var) * ((X-center[0])**2 + (Y-center[1])**2)

    def gaussian(parameters):
        center, covariance, rho = parameters
        g00, g11 = covariance
        g01 = g00*g11*rho
        x, y = X-center[0], Y-center[1]
        det = jnp.abs(g00*g11-g01*g01)
        return -0.5/det * (x**2*g11-x*y*(g01+g01)+y**2*g00)

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


class GeomMaskOperator(ift.LinearOperator):
    """
    Copyright @ Jakob Roth

    Takes a field and extracts the central part of the field corresponding to target.shape

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    target : Domain, DomainTuple or tuple of Domain
        The operator's target domain
    """

    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        sl = []
        for i in range(len(self._domain.shape)):
            slStart = int((self._domain.shape[i] - self._target.shape[i]) / 2.)
            slStop = slStart + self._target.shape[i]
            sl.append(slice(slStart, slStop, 1))
        self._slices = tuple(sl)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._slices]
            return ift.Field(self.target, res)
        res = np.zeros(self.domain.shape, x.dtype)
        res[self._slices] = x
        return ift.Field(self.domain, res)