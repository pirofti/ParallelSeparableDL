# Copyright (c) 2020 Cristi Rusu <cristi.rusu.tgm@gmail.com>
# Copyright (c) 2020 Paul Irofti <paul@irofti.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# import matplotlib.pyplot as plt
import numpy as np
import timeit
from sklearn.linear_model import OrthogonalMatchingPursuit
from skimage.util.shape import view_as_blocks
from skimage import color
from scipy.fftpack import dct
from skimage.io import imread
#############################################################################
n_components = 8      # number of atoms (n)
n_features = 8         # signal dimension (m)
n_nonzeros = 8    # sparsity (s)
n_iterations = 30      # number of dictionary learning iterations (K)
#images = ['images/lena.png', 'images/houses.png', 'images/peppers.png']
images = ['images/lena.png', 'images/houses.png', 'images/peppers.png',
          'images/boat.png', 'images/barbara.png', 'images/pirate.png',
          'images/street.png', 'images/lighthouse.png', 'images/couple.png']
#############################################################################


def odct(m, n):
    """Overcomplete Discrete Cosine Transform"""
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.cos(np.pi*(i+0.5)*j/n)
    for j in range(n):
        if j >= 1:
            D[:, j] = D[:, j] - np.mean(D[:, j])
        D[:, j] = D[:, j]/np.linalg.norm(D[:, j])

    return D


def omp_2D(dictionary, samples, n_nonzero_coefs, params=[]):
    """2D Orthogonal Matching Pursuit"""
    ompfun = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False, normalize=False,
                                       precompute=True)

    samples_vec = np.zeros((samples.shape[1]*samples.shape[2], samples.shape[0]))
    for i in range(samples.shape[0]):
        samples_vec[:, i] = samples[i, :, :].T.reshape(samples.shape[1]*samples.shape[2])

    dictionary_vec = np.kron(dictionary[1], dictionary[0])

    codes_vec = ompfun.fit(dictionary_vec, samples_vec).coef_.T
    #codes_vec = np.zeros((dictionary[0].shape[1]*dictionary[1].shape[1], samples.shape[0]))
    #for i in range(samples.shape[0]):
    #    codes_vec[:, i] = ompfun.fit(dictionary_vec, samples[i, :, :].T.reshape((samples.shape[1]*samples.shape[2]))).coef_.T

    err = np.linalg.norm(samples_vec - dictionary_vec @ codes_vec, 'fro') ** 2

    codes = np.zeros((samples.shape[0], dictionary[0].shape[1], dictionary[1].shape[1]))
    for i in range(samples.shape[0]):
        codes[i, :, :] = codes_vec[:, i].reshape((dictionary[0].shape[1], dictionary[1].shape[1])).T

    return codes, err


# def omp_2D(dictionary, samples, n_nonzero_coefs, params=[]):
#     """2D Orthogonal Matching Pursuit"""
#     ompfun = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False, normalize=False,
#                                        precompute=True)
#     samples_vec = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2]).T
#     dictionary_vec = np.kron(dictionary[1], dictionary[0])
#     codes_vec = ompfun.fit(dictionary_vec, samples_vec).coef_.T
#     err = np.linalg.norm(samples_vec - dictionary_vec @ codes_vec, 'fro') ** 2
#     codes = codes_vec.T.reshape(samples.shape[0], dictionary[0].shape[1], dictionary[1].shape[1])
#
#     return codes, err


#print(len(images))
# Data
start_time = timeit.default_timer()
total_samples = []
for i in range(len(images)):
    samples = imread(images[i], as_gray=True)
    size_x = samples.shape[0]
    size_y = samples.shape[1]
    samples = view_as_blocks(samples, block_shape=(n_features, n_features))
    # (n_samples, n_features, n_features)
    samples = samples.reshape(int(size_x*size_y/n_features**2), n_features, n_features)
    for m in range(samples.shape[0]):
        samples[m] = samples[m] - samples[m].mean()

    if i == 0:
        total_samples = samples
    else:
        total_samples = np.concatenate((total_samples, samples), axis=0)

del(samples)
#print(total_samples.shape)

n_samples = total_samples.shape[0]

D1 = odct(n_features, n_components)
D2 = odct(n_features, n_components)
errs = np.zeros(n_iterations + 1)
codes, err = omp_2D([D1, D2], total_samples, n_nonzeros)
errs[0] = err

for iter in range(n_iterations):
    # update useful matrices
    XXT = np.zeros((D1.shape[1], D1.shape[1]))
    XYT = np.zeros((D1.shape[1], total_samples.shape[1]))
    for m in range(n_samples):
        aux = codes[m, :, :] @ D2.T
        XXT = XXT + aux @ aux.T
        XYT = XYT + aux @ total_samples[m, :, :].T

    # update D1
    D1 = np.linalg.lstsq(XXT, XYT, rcond=-1)[0].T
    for j in range(n_components):
       aux = np.linalg.norm(D1[:, j])
       if aux < 10e-20:
           D1[:, j] = total_samples[int(round(np.random.random()*n_samples)), :, :].reshape(D1.shape[0]**2)
           aux = np.linalg.norm(D1[:, j])
       D1[:, j] = D1[:, j] / aux

    XTX = np.zeros((D2.shape[1], D2.shape[1]))
    XTY = np.zeros((D2.shape[1], total_samples.shape[1]))
    for m in range(n_samples):
        aux = D1 @ codes[m, :, :]
        XTX = XTX + aux.T @ aux
        XTY = XTY + aux.T @ total_samples[m, :, :]

    # update D2
    D2 = np.linalg.lstsq(XTX, XTY, rcond=None)[0].T
    for j in range(n_components):
       aux = np.linalg.norm(D2[:, j])
       if aux < 10e-20:
           D2[:, j] = total_samples[int(round(np.random.random() * n_samples)), :, :].reshape(D2.shape[0] ** 2)
           aux = np.linalg.norm(D2[:, j])
       D2[:, j] = D2[:, j] / aux

    # update representations and error
    codes, err = omp_2D([D1, D2], total_samples, n_nonzeros)
    errs[iter+1] = err#/total_samples.shape[0]

elapsed = timeit.default_timer() - start_time
print("elapsed", elapsed)

# Results
# plt.title("RMSE evolution")
# plt.plot(range(n_iterations+1), errs)
# plt.show()
