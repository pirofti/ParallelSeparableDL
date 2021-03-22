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
from omp import omp, omp_2D
from aksvd import aksvd
from sepsum import sepsumvec_ortho
from htr import htr_2D
from dictionary_learning import dictionary_learning
from skimage.util.shape import view_as_blocks
from skimage.io import imread
from scipy.fftpack import dct
import timeit
import glob
import sys
import pickle
#############################################################################
n_components = 8      # number of atoms (n)
n_features = n_components         # signal dimension (m)
n_nonzeros = 8    # sparsity (s)
n_iterations = 30      # number of dictionary learning iterations (K)
#images = ['images/lena.png', 'images/houses.png', 'images/peppers.png',
#          'images/boat.png', 'images/barbara.png', 'images/pirate.png',
#          'images/street.png', 'images/lighthouse.png', 'images/couple.png']
images = glob.glob('images/*.png')
#############################################################################

def save_test(n_components, n_procs, errs):
    fname='sepsum'
    with open('data/{0}-n{1}-p{2}.dat'.format(fname, n_components, n_procs), 'wb') as fp:
        pickle.dump(errs, fp)

# user input?
if len(sys.argv) > 1:
    n_components = int(sys.argv[1])
    n_features = int(sys.argv[2])

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

Q1 = dct(np.eye(n_components), norm='ortho', axis=0)
Q2 = dct(np.eye(n_components), norm='ortho', axis=0)
codes = np.zeros((total_samples.shape[0], Q1.shape[0], Q2.shape[0]))
for m in range(n_samples):
    aux = Q1.T @ total_samples[m] @ Q2
    aux = aux.reshape(aux.size)
    ind_sort = np.argsort(np.abs(aux))[::-1]
    aux[ind_sort[n_nonzeros + 1:]] = 0
    codes[m] = aux.reshape(codes.shape[1], codes.shape[2])

errs = np.zeros(n_iterations + 1, dtype=float)
err = 0.0
for m in range(n_samples):
    err += np.linalg.norm(total_samples[m] - Q1 @ codes[m] @ Q2.T, 'fro')
errs[0] = err#/total_samples.shape[0]

for iter in range(n_iterations):
    # update useful matrices
    XXT = np.zeros((Q1.shape[0], Q1.shape[0]))
    XYT = np.zeros((Q1.shape[0], total_samples.shape[1]))
    for m in range(n_samples):
        XXT = XXT + codes[m] @ codes[m].T
        XYT = XYT + codes[m] @ Q2.T @ total_samples[m].T

    # update Q1
    Z = XYT.T @ XXT
    U, _, V = np.linalg.svd(Z)
    Q1 = U @ V

    # print(Q1)

    XTX = np.zeros((Q2.shape[0], Q2.shape[0]))
    XTY = np.zeros((Q2.shape[0], total_samples.shape[2]))
    for m in range(n_samples):
        XTX = XTX + codes[m].T @ codes[m]
        XTY = XTY + codes[m].T @ Q1.T @ total_samples[m]

    # update Q2
    Z = XTY.T @ XTX
    U, _, V = np.linalg.svd(Z)
    Q2 = U @ V

    # update representations
    for m in range(n_samples):
        aux = Q1.T @ total_samples[m] @ Q2
        aux = aux.reshape(aux.size)
        ind_sort = np.argsort(np.abs(aux))[::-1]
        aux[ind_sort[n_nonzeros + 1:]] = 0
        codes[m] = aux.reshape(codes.shape[1], codes.shape[2])

    # update error
    err = 0.0
    for m in range(n_samples):
        err += np.linalg.norm(total_samples[m] - Q1 @ codes[m] @ Q2.T, 'fro')
    errs[iter+1] = err#/total_samples.shape[0]

elapsed = timeit.default_timer() - start_time
print("elapsed", elapsed)

# Results
#plt.title("RMSE evolution")
#plt.plot(range(n_iterations+1), errs)
#plt.show()
save_test(n_components, 1, errs)
