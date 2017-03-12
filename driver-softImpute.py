import random
import numpy as np
import scipy.sparse as sp
import softImpute_alg

k = 40  # lower rank
Lambda = 10  # regularization parameter (the same for U and V)
num_iter = 20  # number of iterations in ALS

# Read movie names and print your own ratings
num_users = 943
num_items = 1682
item_names = []
with open('ml-100k/u.item') as f:
	for line in f:
		item_names.append(line.split('|')[1])

# Read train and test data, and create R_train and R_test in CSR sparse format
def read_file(filename):
	raw_data = np.genfromtxt(filename, dtype=np.int32)
	users = raw_data[:, 0] - 1
	items = raw_data[:, 1] - 1
	ratings = raw_data[:, 2].astype(np.float64)
	R = sp.coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
	return R.tocsr()
R_train = read_file('ml-100k/u2.base')
R_test = read_file('ml-100k/u2.test')

U = np.zeros((num_users,k), order='F')
V = np.zeros((num_items,k), order='F')
Dsq = np.eye(k)
U[:] = np.random.normal(num_users*k)

# Call functions in cf for running ALS
print('==== Start running ALS ==== ({0:d} iterations, Lambda={1:g})'.format(num_iter, Lambda))
softImpute_alg.Ssimpute_ALS(R_train, U, V, Dsq, k, 1e-05, Lambda, 20)
print('RMSE on test data:', softImpute_alg.compute_rmse_UV(R_test, U, V))
print('')
