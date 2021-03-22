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
        print(i,j,q.shape)
        #plt.subplot(rows, cols, i*rows + j+1)
        ax1 = plt.subplot(gs1[i*rows + j])
        plt.imshow(q, cmap=plt.cm.gray)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
plt.show()
fig.savefig("atom-pairs.pdf", bbox_inches='tight')