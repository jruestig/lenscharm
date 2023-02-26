import nifty8 as ift
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator

import cluster_fits as cf
from sys import exit


domain = ift.RGSpace((128,)*2, 1.)
space = cf.Space((128*2,)*2, .5)
sampling_points = np.array(space.xycoords.reshape(2, -1))
sampling_points += sampling_points.max()
n_dim, N_points = sampling_points.shape


ndim = sampling_points.shape[0]
mg = np.mgrid[(slice(0, 2),)*ndim]
mg = np.array(list(map(np.ravel, mg)))
dist = domain.distances
dist = np.array(dist).reshape(-1, 1)

pos = sampling_points/dist
excess = pos - np.floor(pos)
pos = np.floor(pos).astype(np.int64)
max_index = np.array(domain.shape).reshape(-1, 1)

if True:
    outside = np.any(
        (pos > max_index) + (pos < 0),
        axis=0
    )
    # pos = pos[:, ~outside]
    # excess = excess[:, ~outside]
    # N_points = (~outside).sum()
else:
    outside = np.full(N_points, False)


data = np.zeros((len(mg[0]), N_points))
datax = np.zeros((len(mg[0]), N_points))
datay = np.zeros((len(mg[0]), N_points))
ii = np.zeros((len(mg[0]), N_points), dtype=np.int64)
jj = np.zeros((len(mg[0]), N_points), dtype=np.int64)

xpre = (-1, -1, 1, 1)
ypre = (-1, 1, -1, 1)
for i in range(len(mg[0])):
    quadrant = np.abs(1 - mg[:, i].reshape(-1, 1) - excess)
    data[i, :] = np.prod(quadrant, axis=0)
    datax[i, :] = xpre[i]*quadrant[1]
    datay[i, :] = ypre[i]*quadrant[0]
    data[i, outside] = 0.
    datax[i, outside] = 0.
    datay[i, outside] = 0.
    fromi = (pos + mg[:, i].reshape(-1, 1)) % max_index
    ii[i, :] = np.arange(N_points)
    jj[i, :] = np.ravel_multi_index(fromi, domain.shape)


mat = coo_matrix((data.reshape(-1), (ii.reshape(-1), jj.reshape(-1))),
                 (N_points, np.prod(domain.shape)))
mat = aslinearoperator(mat)

maty = coo_matrix((datax.reshape(-1), (ii.reshape(-1), jj.reshape(-1))),
        (N_points, np.prod(domain.shape)))
maty = aslinearoperator(maty)

matx = coo_matrix((datay.reshape(-1), (ii.reshape(-1), jj.reshape(-1))),
        (N_points, np.prod(domain.shape)))
matx = aslinearoperator(matx)


# x = np.ones(domain.shape)
# interpolated = mat.matvec(x.reshape(-1))
# 
# plt.imshow(interpolated.reshape(space.shape))
# plt.show()


domainspace = cf.Space(domain.shape, domain.distances)
xx = np.hypot(*domainspace.xycoords)
yy = np.hypot(*space.xycoords)


interpolated = mat.matvec(xx.reshape(-1)).reshape(space.shape)
dxinterpolated = matx.matvec(xx.reshape(-1)).reshape(space.shape)
dyinterpolated = maty.matvec(xx.reshape(-1)).reshape(space.shape)


fig, ((a, b, c), (d, e, f)) = plt.subplots(2, 3)
a.imshow(np.gradient(interpolated, 0.5)[0])
b.imshow(dxinterpolated)
c.imshow(np.gradient(interpolated, 0.5)[0]-dxinterpolated)
d.imshow(np.gradient(interpolated, 0.5)[1])
e.imshow(dyinterpolated)
f.imshow(np.gradient(interpolated, 0.5)[1]-dyinterpolated)
plt.show()


# inter = ift.LinearInterpolator(
#     domain, np.array(space.xycoords.reshape(2, -1)))
# intercut = ift.LinearInterpolator(
#     domain, np.array(space.xycoords.reshape(2, -1)), cast_to_zero=True)

# field = ift.Field.from_raw(domain, np.ones(domain.shape))

# ifield = inter(field).val.reshape((256,)*2)
# cfield = intercut(field).val.reshape((256,)*2)
