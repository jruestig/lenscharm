from functools import reduce
from nifty8.operators.operator import add

import nifty8 as ift
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator


def _compute_FD_gradients(points, field):
    shp = points.shape
    f = field.val
    ndim = shp[0]
    dist = list(field.domain[0].distances)
    dist = np.array(dist).reshape(-1,1)
    pos = points/dist
    excess = pos - np.floor(pos)
    pos = np.floor(pos).astype(np.int64)
    max_index = np.array(field.domain.shape).reshape(-1,1)
    if False:
        outside = np.any(
            (pos > max_index) + (pos < 0),
            axis=0
        )
        print('Casting {} points to 0'.format(outside.sum()))

    data = np.zeros(shp)
    mg = np.mgrid[(slice(0,2),)*ndim]
    mg = np.array(list(map(np.ravel, mg)))

    for axis in range(ndim):
        dx = field.domain[0].distances[axis]
        for i in range(len(mg[0])):
            factor = np.abs(1 - mg[:, i].reshape(-1, 1) - excess)
            factor[axis, :] = (mg[axis, i]*2-1) / dx
            if False:
                factor[axis, outside] = 0.
            indx = (pos+mg[:, i].reshape(-1, 1))
            indx[0, :] = indx[0, :] % field.shape[0]
            indx[1, :] = indx[1, :] % field.shape[1]
            val = np.prod(factor, axis=0)*f[tuple(indx)]
            data[axis, :] += val
    return data


class PartialVdotOperator(ift.LinearOperator):
    def __init__(self, mat):
        self._domain = mat.domain
        shp = mat.domain.shape
        self._target = ift.makeDomain(ift.UnstructuredDomain(shp[1]))
        self._mat = mat.val
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val
        if mode == self.TIMES:
            val = np.sum(self._mat*v, axis=0)
            return ift.Field.from_raw(self._target, val)
        else:
            val = self._mat*v
            return ift.Field.from_raw(self._domain, val)


class PeriodicBoundary(ift.Operator):
    """
    Implementing the periodic boundary positions of an RGSpace for
    pixel coordinates in an unstructured domain

    Parameters
    ----------
    domain_and_target : Unstructured domain in of the pixel coordinates
    RG_domain : RGSpace for which the periodic boundary conditions are implemented

    """
    def __init__(self, domain_and_target, RG_domain):
        self._domain = ift.makeDomain(domain_and_target)
        self._target = ift.makeDomain(domain_and_target)
        self._RG = ift.makeDomain(RG_domain)

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        xval = x
        if lin:
            xval = x.val
        pts = xval.val_rw()
        d = np.array(self._RG[0].distances)
        #dist = (np.array(self._RG[0].shape) - 1) * d
        dist = (np.array(self._RG[0].shape)) * d
        shp = pts.shape
        for axis in range(0,shp[0]):
            pts[axis,:] = pts[axis,:] % dist[axis]
        xval = ift.Field.from_raw(self._target, pts)
        if not lin:
            return xval
        else:
            return x.new(xval, (x.jac))


class Interpolation(ift.Operator):
    """
    Multilinear interpolation for variable points in an RGSpace

    Parameters
    ----------
    rg_dom : RGSpace domain
    rg_key : key of RGSpace in multifield dictionary
    points_dom: domain of interpolation points (unstructured)
    point_key: key of points domain in multifield dictionary
    point_op : Operator
    """
    def __init__(self, rg_dom, rg_key, point_dom, point_key):
        if not isinstance(rg_dom, ift.RGSpace):
            raise TypeError
        if not isinstance(point_dom, ift.UnstructuredDomain):
            raise TypeError
        shp = point_dom.shape
        if len(shp) != 2:
            raise ValueError('Point domain shape length incompatible')
        if shp[0] != len(rg_dom.shape):
            raise ValueError('Point domain incompatible with RGSpace')
        self._domain = ift.MultiDomain.make(
            {rg_key: rg_dom, point_key: point_dom})
        self._target = ift.makeDomain(ift.UnstructuredDomain(shp[1]))
        self._rg_key = rg_key
        self._point_key = point_key

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        xval = x
        if lin:
            xval = x.val
        ps = xval[self._point_key]
        gd = xval[self._rg_key]
        ps = ps.val
        gd = gd.val
        LI = NiftyLinearInterpolator(self._domain[self._rg_key], ps, cast_to_zero=True)
        if not lin:
            return LI(x[self._rg_key])
        val = LI(xval[self._rg_key])
        d1 = LI.ducktape(self._rg_key)

        dat = _compute_FD_gradients(ps, xval[self._rg_key])
        grad = ift.Field.from_raw(x.target[self._point_key], dat)
        d2 = PartialVdotOperator(grad).ducktape(self._point_key)
        return x.new(val, (d1+d2)(x.jac))


class Transponator(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # self._check_input(x, mode)
        return ift.makeField(self._domain, x.val.T)


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


class NiftyLinearInterpolator(ift.LinearOperator):
    """Multilinear interpolation for points in an RGSpace

    Parameters
    ----------
    domain : RGSpace
    sampling_points : numpy.ndarray
        Positions at which to interpolate, shape (dim, ndata),

    Notes
    -----
    Positions that are not within the RGSpace are wrapped according to
    periodic boundary conditions. This reflects the general property of
    RGSpaces to be tori topologically.
    """
    def __init__(self, domain, sampling_points, cast_to_zero=False):
        self._domain = ift.makeDomain(domain)
        for dom in self.domain:
            if not isinstance(dom, ift.RGSpace):
                raise TypeError
        dims = [len(dom.shape) for dom in self.domain]

        # FIXME This needs to be removed as soon as the bug below is fixed.
        if dims.count(dims[0]) != len(dims):
            raise TypeError("This is a bug. Please extend"
                            "LinearInterpolator's functionality!")

        shp = sampling_points.shape
        if not (isinstance(sampling_points, np.ndarray) and len(shp) == 2):
            raise TypeError
        n_dim, n_points = shp
        if n_dim != reduce(add, dims):
            raise TypeError
        self._target = ift.makeDomain(ift.UnstructuredDomain(n_points))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._build_mat(sampling_points, n_points, cast_to_zero)

    def _build_mat(self, sampling_points, N_points, cast_to_zero):
        ndim = sampling_points.shape[0]
        mg = np.mgrid[(slice(0, 2),)*ndim]
        mg = np.array(list(map(np.ravel, mg)))
        dist = [list(dom.distances) for dom in self.domain]
        # FIXME This breaks as soon as not all domains have the same number of
        # dimensions.
        dist = np.array(dist).reshape(-1, 1)
        pos = sampling_points/dist
        excess = pos - np.floor(pos)
        pos = np.floor(pos).astype(np.int64)
        max_index = np.array(self.domain.shape).reshape(-1, 1)
        if cast_to_zero:
            outside = np.any(
                (pos > max_index) + (pos < 0),
                axis=0
            )
            # print('Casting {} points to 0'.format(outside.sum()))
        else:
            outside = np.full(N_points, False)
        data = np.zeros((len(mg[0]), N_points))
        ii = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        jj = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        for i in range(len(mg[0])):
            factor = np.prod(
                np.abs(1 - mg[:, i].reshape(-1, 1) - excess), axis=0)
            data[i, :] = factor
            data[i, outside] = 0.
            fromi = (pos + mg[:, i].reshape(-1, 1)) % max_index
            ii[i, :] = np.arange(N_points)
            jj[i, :] = np.ravel_multi_index(fromi, self.domain.shape)
        self._mat = coo_matrix((data.reshape(-1),
                                (ii.reshape(-1), jj.reshape(-1))),
                               (N_points, np.prod(self.domain.shape)))
        self._mat = aslinearoperator(self._mat)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_val = x.val
        if mode == self.TIMES:
            res = self._mat.matvec(x_val.reshape(-1))
        else:
            res = self._mat.rmatvec(x_val).reshape(self.domain.shape)
        return ift.Field(self._tgt(mode), res)
