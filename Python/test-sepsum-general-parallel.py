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
import multiprocessing
import numpy as np
import timeit
from sklearn.linear_model import OrthogonalMatchingPursuit
from skimage.util.shape import view_as_blocks
from skimage import color
from skimage.io import imread
import glob
import sys
import pickle
#############################################################################
n_components = 8  # number of atoms (n)
n_features = 8  # signal dimension (m)
n_nonzeros = 16  # sparsity (s)
n_iterations = 100  # number of dictionary learning iterations (K)
#images = ['images/lena.png', 'images/houses.png', 'images/peppers.png']
#images = ['images/lena.png', 'images/houses.png', 'images/peppers.png',
#          'images/boat.png', 'images/barbara.png', 'images/pirate.png',
#          'images/street.png', 'images/lighthouse.png', 'images/couple.png']
images = glob.glob('images/*.png')

PROC_NUM = 3   # how many processes to spawn
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


def process_data(queue1, queue2, barrier, images, n_features, n_nonzeros, n_iterations):
    # Data
    #print("Processing:", images)
    total_samples = []
    for i in range(len(images)):
        # print("Reading", images[i])
        samples = imread(images[i], as_gray=True)
        size_x = samples.shape[0]
        size_y = samples.shape[1]
        samples = view_as_blocks(samples, block_shape=(n_features, n_features))
        # (n_samples, n_features, n_features)
        samples = samples.reshape(int(size_x * size_y / n_features ** 2), n_features, n_features)
        # print(samples.max())
        for m in range(samples.shape[0]):
            samples[m] = samples[m] - samples[m].mean()
        if i == 0:
            total_samples = samples
        else:
            total_samples = np.concatenate((total_samples, samples), axis=0)

    del (samples)
    # print(total_samples.shape)

    n_samples = total_samples.shape[0]
    D1 = odct(n_features, n_components)
    D2 = odct(n_features, n_components)
    codes, err = omp_2D([D1, D2], total_samples, n_nonzeros)
    # send the error to master
    queue1.put(err)

    # get confirmation to continue
    # valuee = queue2.get()
    barrier.wait()

    # main loop
    for iter in range(n_iterations):
        # update useful matrices
        XXT = np.zeros((D1.shape[1], D1.shape[1]))
        XYT = np.zeros((D1.shape[1], total_samples.shape[1]))
        for m in range(n_samples):
            aux = codes[m, :, :] @ D2.T
            XXT = XXT + aux @ aux.T
            XYT = XYT + aux @ total_samples[m, :, :].T
        # send partial results for the calculation of D1
        barrier.wait()
        queue1.put([XXT, XYT])

        # get new D1
        D1 = queue2.get()

        # update useful matrices
        XTX = np.zeros((D2.shape[1], D2.shape[1]))
        XTY = np.zeros((D2.shape[1], total_samples.shape[1]))
        for m in range(n_samples):
            aux = D1 @ codes[m, :, :]
            XTX = XTX + aux.T @ aux
            XTY = XTY + aux.T @ total_samples[m, :, :]
        # send partial results for the calculation of D2
        queue1.put([XTX, XTY])
        barrier.wait()

        #  get new D2
        D2 = queue2.get()

        # update representations and error
        codes, err = omp_2D([D1, D2], total_samples, n_nonzeros)
        # send the error to master
        queue1.put(err)

        # get confirmation to continue
        # valuee = queue2.get()
        barrier.wait()

def save_test(n_components, n_procs, errs):
    fname='sepsum-general'
    with open('data/{0}-n{1}-p{2}.dat'.format(fname, n_components, n_procs), 'wb') as fp:
        pickle.dump(errs, fp)

if __name__ == '__main__':
    #print("Number of cpus:", multiprocessing.cpu_count())
    start_time = timeit.default_timer()
    # send data from master to workers
    queue1 = multiprocessing.Queue()
    # send data from workers to master
    queue2 = multiprocessing.Queue()

    # user input?
    if len(sys.argv) > 1:
        PROC_NUM = int(sys.argv[1])
        n_components = int(sys.argv[2])
        n_features = int(sys.argv[3])

    # we cannot have more processes than images
    PROC_NUM = np.min([PROC_NUM, len(images)])
    barrier = multiprocessing.Barrier(PROC_NUM)

    # distribute images among processes as equitable as possible
    images_per_process = np.ones(PROC_NUM, dtype=int) * int(np.ceil(len(images) / PROC_NUM))
    diff = PROC_NUM * int(np.ceil(len(images) / PROC_NUM)) - len(images)
    index = len(images_per_process) - 1
    while diff > 0:
        images_per_process[index] -= 1
        diff -= 1
        index -= 1
    grid = [0]
    for val in images_per_process:
        grid.append(grid[-1] + val)
    #print(grid)

    processes = []
    for i in range(len(grid) - 1):
        # print(images[grid[i]:grid[i + 1]])
        t = multiprocessing.Process(target=process_data, args=(
            queue1, queue2, barrier, images[grid[i]:grid[i + 1]],
            n_features, n_nonzeros, n_iterations))
        t.start()
        processes.append(t)

    errs = np.zeros(n_iterations + 1, dtype=float)
    # collect initialization errors
    values = []
    for i in range(PROC_NUM):
        values.append(queue1.get())
    # print(values)
    errs[0] = np.sum(values)

    # and then send confirmation to continue
    # for i in range(PROC_NUM):
    #    queue2.put(1)

    # main loop
    for iter in range(n_iterations):
        XXT = np.zeros((n_components, n_components))
        XYT = np.zeros((n_components, n_features))
        for i in range(PROC_NUM):
            aux = queue1.get()
            XXT += aux[0]
            XYT += aux[1]

        # update D1
        D1 = np.linalg.lstsq(XXT, XYT, rcond=-1)[0].T
        for j in range(n_components):
            aux = np.linalg.norm(D1[:, j])
            if aux < 10e-20:
                D1[:, j] = total_samples[int(round(np.random.random() * n_samples)), :, :].reshape(D1.shape[0] ** 2)
                aux = np.linalg.norm(D1[:, j])
            D1[:, j] = D1[:, j] / aux
        # send new D1 to everyone
        for i in range(PROC_NUM):
            queue2.put(D1)

        XTX = np.zeros((n_components, n_components))
        XTY = np.zeros((n_components, n_features))
        for i in range(PROC_NUM):
            aux = queue1.get()
            XTX += aux[0]
            XTY += aux[1]

        # update D2
        D2 = np.linalg.lstsq(XTX, XTY, rcond=None)[0].T
        for j in range(n_components):
            aux = np.linalg.norm(D2[:, j])
            if aux < 10e-20:
                D2[:, j] = total_samples[int(round(np.random.random() * n_samples)), :, :].reshape(D2.shape[0] ** 2)
                aux = np.linalg.norm(D2[:, j])
            D2[:, j] = D2[:, j] / aux
        # send new D2 to everyone
        for i in range(PROC_NUM):
            queue2.put(D2)

        # collect initialization errors
        values = []
        for i in range(PROC_NUM):
            values.append(queue1.get())
        errs[iter + 1] = np.sum(values)

        # and then send confirmation to continue
        # for i in range(PROC_NUM):
        #     queue2.put(1)

    # wrap up processes
    for proc in processes:
        proc.join()

    elapsed = timeit.default_timer() - start_time
    print("elapsed", elapsed)

    # Results
    # plt.title("RMSE evolution")
    # plt.plot(range(n_iterations + 1), errs)
    # plt.show()
    save_test(n_components, PROC_NUM, errs)
