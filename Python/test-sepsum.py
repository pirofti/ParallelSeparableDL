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


#############################################################################
import matplotlib.pyplot as plt
import numpy as np
from omp import omp, omp_2D
from aksvd import aksvd
from sepsum import sepsumvec_ortho
from htr import htr_2D
from dictionary_learning import dictionary_learning
from skimage.util.shape import view_as_blocks
from skimage import color
from scipy.fftpack import dct
#############################################################################
n_components = 8      # number of atoms (n)
n_features = 8         # signal dimension (m)
n_nonzero_coefs = 6    # sparsity (s)
n_samples = 4096        # number of signals (N)
n_iterations = 20      # number of dictionary learning iterations (K)
image = 'images/lena.bmp'
#############################################################################

# Data
samples = color.rgb2gray(plt.imread(image))
samples = samples - samples.mean()
# Fetch distinct patches
samples = view_as_blocks(samples, block_shape=(n_features, n_features))
# (n_samples, n_features, n_features)
samples = samples.reshape(-1, n_features, n_features)[::8]
# Extract n_samples at random
samples = samples[np.random.choice(samples.shape[0], n_samples)]

D1 = dct(np.eye(n_components), norm='ortho', axis=0)
D2 = dct(np.eye(n_components), norm='ortho', axis=0)
# D1 = np.random.standard_normal((n_features, n_components))
# D2 = np.random.standard_normal((n_features, n_components))
dictionary = [D1, D2]


# distort the clean signal
samples = samples + 0.05 * np.random.standard_normal(samples.shape)

# AK-SVD
D = np.kron(D1, D2)
D = np.append(D,D, axis=1)
Y = samples.reshape(samples.shape[0], samples.shape[1]*samples.shape[2]).T
dict_ak, codes_ak, rmse_ak = dictionary_learning(Y, D, n_nonzero_coefs,
                                                 n_iterations, omp, aksvd)
# SEPSUM
dict_sepsum, codes_sepsum, rmse_sso = dictionary_learning(
    samples, dictionary, n_nonzero_coefs, n_iterations,
    htr_2D, sepsumvec_ortho)


# Results
plt.title("RMSE evolution")
plt.plot(range(n_iterations), rmse_ak)
plt.plot(range(n_iterations), rmse_sso)
plt.show()
