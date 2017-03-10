import random
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class DataVisualization:
    """ PCA-based visualization on MovieLens data sets """

    def __init__(self, txt_data, M, N):
        """ Build dense matrix out of data from text file """

        raw_data = np.genfromtxt(txt_data, dtype=np.int32)
        item_ids = raw_data[:, 0] - 1
        comp_ids = raw_data[:, 1] - 1
        values = raw_data[:, 2].astype(np.float64)

        #convenient to initially build sparse matrix but will need dense later
        R = sp.coo_matrix((values, (comp_ids, item_ids)), shape=(M, N))
        self._data = np.zeros((M, N), order='F')
        self._data[:, :] = R.toarray()
        self._M = M
        self._N = N


    def gen_plot(self, figname, special=[]):
        """
            Generate low-dimensional plot of input txt_data
            Gives different marker to ids in special list
        """
        mean = self._data.mean(axis=1)
        self._data = self._data - mean.reshape((self._M, 1))
        U, _, _ = np.linalg.svd(self._data, full_matrices=False)
        princ_comps = U.T.dot(self._data)

        # Generate Plot
        plt.figure()
        for i in xrange(self._N):
            if i in special:
                continue
            plt.plot(princ_comps[0,i], princ_comps[1,i], 'sb')

        # Give different marker to those in ignore list
        for i in special:
            plt.plot(princ_comps[0, i], princ_comps[1,i], '^g')

        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

        plt.savefig(figname)

# use module to plot MovieLens 100k data set only if directly called
# otherwise just provide class interface
def main():
    dv = DataVisualization('data/ml-100k/u.data', 1682, 943)
    dv.gen_plot('plots/init100k.pdf', special=[5,9])


if __name__ == "__main__":
    main()
