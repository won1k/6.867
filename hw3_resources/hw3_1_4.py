import numpy as np
import nn
from plotBoundary import *
import pylab as pl

# Load data
n = 1
data = np.genfromtxt('data/data' + str(n) + '_train.csv', delimiter = ' ')
valid = np.genfromtxt('data/data' + str(n) + '_validate.csv', delimiter = ' ')
test = np.genfromtxt('data/data' + str(n) + '_test.csv', delimiter = ' ')
X_train = data[:,0:2]
Y_train = [int(y) for y in data[:,-1]]
X_valid = valid[:,0:2]
Y_valid = [int(y) for y in valid[:,-1]]
X_test = test[:,0:2]
Y_test = [int(y) for y in test[:,-1]]
n = len(X_train)

# Initialize NN/loss
net = nn.NN(2, 5, 3, 1)

# Train NN
epochs = 1000
lamb = 0.005
prev_loss = 10e5
curr_loss = prev_loss-1
while abs(curr_loss - prev_loss) > 1e-10:
	# SGD
	perm_order = np.random.permutation(n)
	for i in perm_order:
		net.update(X_train[i,:], Y_train[i], lamb)
	# Loss
	prev_loss = curr_loss
	curr_loss = net.updateLoss(X_valid,Y_valid)
	print "Validation loss: " + str(curr_loss)

# Test loss
print "Test loss: " + str(net.updateLoss(X_test, Y_test))

# Plot
plotDecisionBoundary(X_test, Y_test, net.predict, [-1,0,1], title = 'Test Set 1')
pl.savefig('hw3_1-4' + str(n) + '.pdf')