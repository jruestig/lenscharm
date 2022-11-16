import cluster_fits as cf
import numpy as np
import matplotlib.pyplot as plt
import h5py

from os.path import join
from source_fwd import Blurring, bigsmoother


space = cf.Space((1024,)*2, 1.)
klcoords = space.klcoords

field = np.array(h5py.File(join('/home/jruestig/Data/angulo/tcm_clusters/', 'big_clusters_adj_{}.h5'.format(0)))['data'])
field = Blurring(field, bigsmoother)

grad = np.gradient(field, 1.)
grad2 = np.fft.ifft2(np.fft.fft2(field)*2j*np.pi*klcoords)

vm = 1e-3
fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(grad[0])
axes[1, 0].imshow(grad[1])
axes[0, 1].imshow(grad2.real[1])
axes[1, 1].imshow(grad2.real[0])
axes[0, 2].imshow((grad[0]-grad2.real[1]), vmax=vm, vmin=-vm)
axes[1, 2].imshow((grad[1]-grad2.real[0]), vmax=vm, vmin=-vm)
# axes[0, 0].suptitle('Numpy grad x')
# axes[1, 0].suptitle('Numpy grad y')
# axes[0, 1].suptitle('FFT grad x')
# axes[1, 1].suptitle('FFT grad y')
plt.show()
