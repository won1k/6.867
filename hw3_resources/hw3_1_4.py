import numpy as np
import nn
from plotBoundary import *
import pylab as pl
import sys

# Hyperparams
n = sys.argv[1] # data
nlayers = int(sys.argv[2])
hdim = int(sys.argv[3])
#print n, nlayers, hdim

# Load data
data = np.genfromtxt('data/data' + n + '_train.csv', delimiter = ' ')
valid = np.genfromtxt('data/data' + n + '_validate.csv', delimiter = ' ')
test = np.genfromtxt('data/data' + n + '_test.csv', delimiter = ' ')
X_train = data[:,0:2]
Y_train = [int(y) for y in data[:,-1]]
X_valid = valid[:,0:2]
Y_valid = [int(y) for y in valid[:,-1]]
X_test = test[:,0:2]
Y_test = [int(y) for y in test[:,-1]]
N = len(X_train)

# Initialize NN/loss
net = nn.NN(2, hdim, 2, nlayers)

# Train NN
max_epochs = 5e3
lamb = 0.1
prev_loss = 10e5
curr_loss = prev_loss-1
epoch = 0
train_loss = []
valid_loss = []
test_loss = []
while abs(curr_loss - prev_loss) > 1e-8 and epoch < max_epochs:
	# SGD
	perm_order = np.random.permutation(N)
	for i in perm_order:
		net.update(X_train[i,:], max(Y_train[i],0), lamb)
	# Loss
	prev_loss = curr_loss
	curr_loss = net.updateLoss(X_valid, [max(y,0) for y in Y_valid])
	#print "Validation loss: " + str(curr_loss), "Learning rate : " + str(lamb), "Epoch :" + str(epoch)
	if (curr_loss - prev_loss)/prev_loss > 1e-2:
		lamb = lamb*0.9
	epoch += 1
	# Keep losses
	train_loss.append(net.updateLoss(X_train, [max(y,0) for y in Y_train]))
	valid_loss.append(net.updateLoss(X_valid, [max(y,0) for y in Y_valid]))
	test_loss.append(net.updateLoss(X_test, [max(y,0) for y in Y_test]))

# Test loss
#print "Test loss: " + str(net.updateLoss(X_test, Y_test))
acc = 0.0
Nt = len(X_test)
for i in range(Nt):
	pred = net.predict(X_test[i,:])
	if pred == Y_test[i]:
		acc += 1
#print "Test accuracy: " + str(acc / Nt)
print "Dataset: ", n, "Num. layers:", nlayers, "Hid. dim:", hdim, "Epochs: ", str(epoch), "Test loss:", str(net.updateLoss(X_test, [max(y,0) for y in Y_test])), "Test accuracy: ", str(acc/Nt)

# Plot boundaries
plotDecisionBoundary(X_train, Y_train, net.predict, [-1,0,1], title = 'Train Set ' + n + ', ' + str(nlayers) + ' Layers, ' + str(hdim) + ' Hid. Dim.')
pl.savefig('hw3_1-4' + '_' +  n + '_' + str(nlayers) + '_' + str(hdim) + '_train.pdf')
plotDecisionBoundary(X_test, Y_test, net.predict, [-1,0,1], title = 'Test Set ' + n)
pl.savefig('hw3_1-4' + '_' + n + '_' + str(nlayers) + '_' + str(hdim) + '_test.pdf')

# Plot loss
pl.figure()
pl.plot(range(epoch), train_loss, label = "Train")
pl.plot(range(epoch), valid_loss, label = "Valid", ls = 'dashed')
pl.plot(range(epoch), test_loss, label = "Test", ls = 'dashed')
pl.xlabel("Epochs")
pl.ylabel("Loss")
pl.legend()
pl.savefig('hw3_1-4' + '_' +  n + '_' + str(nlayers) + '_' + str(hdim) + '_loss.pdf')
