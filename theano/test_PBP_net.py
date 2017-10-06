import matplotlib.pyplot as plt
import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net
import time

np.random.seed(1)

# We load the boston housing dataset

data = np.loadtxt('boston_housing.txt')

# We obtain the features and the targets

X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

# We create the train and test sets with 90% and 10% of the data

permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = int(np.round(X.shape[ 0 ] * 0.9))
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test = X[ index_test, : ]
y_test = y[ index_test ]

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

t1 = time.time()
n_hidden_units = 10
net = PBP_net.PBP_net(X_train, y_train,
    [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 10)
print("took %f seconds to finish training" % (time.time() - t1))

# We make predictions for the test set

# m, v, v_noise = net.predict(X_test)
# m, v, v_noise = net.predict(X_train)

m, v, v_noise = net.predict(X_test)
m_train, v_train, v_noise_train = net.predict(X_train)
# We compute the test RMSE

# rmse = np.sqrt(np.mean((y_test - m)**2))
rmse = np.sqrt(np.mean((y_train - m_train)**2))

print rmse

# We compute the test log-likelihood

# test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
#     0.5 * (y_test - m)**2 / (v + v_noise))
test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v_train + v_noise)) - \
    0.5 * (y_train - m_train)**2 / (v_train + v_noise_train))
print test_ll


order = np.argsort(y_test)
order_train = np.argsort(y_train)


plt.plot(y_test[order], (m + np.sqrt(v))[order], c='b')
plt.plot(y_test[order], (m - np.sqrt(v))[order], c='b')
plt.scatter(y_test, m, c='r')
plt.scatter(y_train, m_train, c='k')
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k')
plt.xlabel("Y")
plt.ylabel("Y_hat")
plt.savefig("theano.png")
