import matplotlib.pyplot as plt
import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net
import time

np.random.seed(1)


N = 200
X_train = np.linspace(0, 50, N) + np.random.normal(scale=1, size=N)
np.random.shuffle(X_train)
y_train = X_train * 5 + 3 + np.random.normal(scale=1, size=N)
X_train = X_train.reshape([-1, 1])


# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

t1 = time.time()
n_hidden_units = 10
print("ready")
net = PBP_net.PBP_net(X_train, y_train,
    [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 10)
print("took %f seconds to finish training" % (time.time() - t1))

# We make predictions for the test set

# m, v, v_noise = net.predict(X_test)
# m, v, v_noise = net.predict(X_train)

X_test = np.linspace(-100, 150, N) + np.random.normal(scale=1, size=N)
np.random.shuffle(X_test)
y_test = X_test * 5 + 3 + np.random.normal(scale=1, size=N)
X_test = X_test.reshape([-1, 1])
size_test = N

m, v, v_noise = net.predict(X_test)
m_train, v_train, v_noise_train = net.predict(X_train)
# We compute the test RMSE

# rmse = np.sqrt(np.mean((y_test - m)**2))
rmse = np.sqrt(np.mean((y_train - m)**2))

print rmse

# We compute the test log-likelihood

# test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
#     0.5 * (y_test - m)**2 / (v + v_noise))
test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_train - m)**2 / (v + v_noise))
print test_ll


order = np.argsort(y_test)
order_train = np.argsort(y_train)


plt.plot(y_test[order], (m + np.sqrt(v))[order], c='b')
plt.plot(y_test[order], (m - np.sqrt(v))[order], c='b')
plt.scatter(y_test, m, c='r')
plt.scatter(y_train, m_train, c='k')
plt.plot([np.min(y_test) - 50, np.max(y_test) + 50], [np.min(y_test) - 50, np.max(y_test) + 50], 'k')
plt.xlabel("Y")
plt.ylabel("Y_hat")
plt.savefig("linear.png")
