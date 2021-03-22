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
from skimage.util.shape import view_as_blocks
from skimage.io import imread
from scipy.fftpack import dct
import timeit
import multiprocessing
import glob
import sys
import pickle
#############################################################################
n_components = 8  # number of atoms (n)
n_features = n_components  # signal dimension (m)
n_nonzeros = 16  # sparsity (s)
n_iterations = 100  # number of dictionary learning iterations (K)
#images = ['images/lena.png', 'images/houses.png', 'images/peppers.png',
#          'images/boat.png', 'images/barbara.png', 'images/pirate.png',
#          'images/street.png', 'images/lighthouse.png', 'images/couple.png']
#images = glob.glob('images/*.png')
images = ['images/lena.png']

PROC_NUM = 8   # how many processes to spawn
#############################################################################


def process_data(queue1, queue2, barrier, images, n_features, n_nonzeros, n_iterations):
    # Data
    # print("Processing:", images)
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
    Q1 = dct(np.eye(n_components), norm='ortho', axis=0)
    Q2 = dct(np.eye(n_components), norm='ortho', axis=0)
    codes = np.zeros((total_samples.shape[0], Q1.shape[0], Q2.shape[0]))
    for m in range(n_samples):
        aux = Q1.T @ total_samples[m] @ Q2
        aux = aux.reshape(aux.size)
        ind_sort = np.argsort(np.abs(aux))[::-1]
        aux[ind_sort[n_nonzeros + 1:]] = 0
        codes[m] = aux.reshape(codes.shape[1], codes.shape[2])

    # errs = np.zeros(n_iterations + 1, dtype=float)
    err = 0.0
    for m in range(n_samples):
        err += np.linalg.norm(total_samples[m] - Q1 @ codes[m] @ Q2.T, 'fro')
    #err = err / total_samples.shape[0]
    # send the error to master
    queue1.put(err)

    # get confirmation to continue
    #valuee = queue2.get()
    barrier.wait()

    # main loop
    for iter in range(n_iterations):
        # update useful matrices
        XXT = np.zeros((Q1.shape[0], Q1.shape[0]))
        XYT = np.zeros((Q1.shape[0], total_samples.shape[1]))
        for m in range(n_samples):
            XXT = XXT + codes[m] @ codes[m].T
            XYT = XYT + codes[m] @ Q2.T @ total_samples[m].T
        Z = XYT.T @ XXT
        # send partial results for the calculation of Q1
        barrier.wait()
        queue1.put(Z)

        # get new Q1
        Q1 = queue2.get()

        # update useful matrices
        XTX = np.zeros((Q2.shape[0], Q2.shape[0]))
        XTY = np.zeros((Q2.shape[0], total_samples.shape[2]))
        for m in range(n_samples):
            XTX = XTX + codes[m].T @ codes[m]
            XTY = XTY + codes[m].T @ Q1.T @ total_samples[m]
        Z = XTY.T @ XTX
        # send partial results for the calculation of Q2
        queue1.put(Z)
        barrier.wait()

        #  get new Q2
        Q2 = queue2.get()

        # update representations
        for m in range(n_samples):
            aux = Q1.T @ total_samples[m] @ Q2
            aux = aux.reshape(aux.size)
            ind_sort = np.argsort(np.abs(aux))[::-1]
            aux[ind_sort[n_nonzeros + 1:]] = 0
            codes[m] = aux.reshape(codes.shape[1], codes.shape[2])

        # compute error
        err = 0.0
        for m in range(n_samples):
            err += np.linalg.norm(total_samples[m] - Q1 @ codes[m] @ Q2.T, 'fro')
        #err = err / total_samples.shape[0]
        # send the error to master
        # print('<PRODUCER>')
        # print(n_samples)
        # print(type(err))
        # print(err)
        # print('</PRODUCER>')
        queue1.put(err)
        queue1.put(codes)

        # get confirmation to continue
        #valuee = queue2.get()
        barrier.wait()

def save_test(n_components, n_procs, errs):
    fname='sepsum'
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
    # print(grid)

    processes = []
    for i in range(len(grid) - 1):
        # print(images[grid[i]:grid[i + 1]])
        t = multiprocessing.Process(target=process_data, args=(
        queue1, queue2, barrier, images[grid[i]:grid[i + 1]], n_features, n_nonzeros, n_iterations))
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
    #for i in range(PROC_NUM):
    #    queue2.put(1)

    # main loop
    for iter in range(n_iterations):
        Z = np.zeros((n_components, n_components))
        for i in range(PROC_NUM):
            aux = queue1.get()
            Z += aux

        # update Q1
        U, _, V = np.linalg.svd(Z)
        Q1 = U @ V
        # send new Q1 to everyone
        for i in range(PROC_NUM):
            queue2.put(Q1)

        Z = np.zeros((n_components, n_components))
        for i in range(PROC_NUM):
            aux = queue1.get()
            Z += aux

        # update Q2
        U, _, V = np.linalg.svd(Z)
        Q2 = U @ V
        # send new Q2 to everyone
        for i in range(PROC_NUM):
            queue2.put(Q2)

        # collect initialization errors
        values.clear()
        # print(values)
        for i in range(PROC_NUM):
            values.append(queue1.get())
            # v = queue1.get()
            # print(i, v)
            # values.append(v)
        # print(type(values))
        # print(values)
        errs[iter + 1] = np.sum(values)
        
        # collect codes
        values.clear()
        for i in range(PROC_NUM):
            values.append(queue1.get())
        codes = values

        # and then send confirmation to continue
        #for i in range(PROC_NUM):
        #    queue2.put(1)

    # wrap up processes
    for proc in processes:
        proc.join()

    elapsed = timeit.default_timer() - start_time
    print("elapsed", elapsed)

    # Results
    #plt.title("RMSE evolution")
    #plt.plot(range(n_iterations + 1), errs)
    #plt.show()
    #save_test(n_components, PROC_NUM, errs)
    codes = codes[0]
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    rows = Q1.shape[0]
    cols = Q2.shape[0]
    
    fig = plt.figure(figsize = (rows,cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
    
    for i in range(rows):
        for j in range(cols):
            q =  np.outer(Q1[:,i],Q2[j,:])
            #print(i,j,q.shape)
            ax1 = plt.subplot(gs1[i*rows + j])
            plt.imshow(q, cmap=plt.cm.gray)
            plt.axis('on')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
    plt.show()
    fig.savefig("atom-pairs.pdf", bbox_inches='tight')