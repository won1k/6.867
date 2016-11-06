import numpy as np
import nn
from plotBoundary import *
import pylab as pl
import sys

# Hyperparams
nlayers = int(sys.argv[1])
hdim = int(sys.argv[2])
#print n, nlayers, hdim

# Load data
digits = range(10)
samples = 300 # samples per digit (half for valid and test)

file = np.loadtxt("data/mnist_digit_1.csv")
X_train = file[0:samples,:]
X_valid = file[samples:(3*samples/2),:]
X_test  = file[(3*samples/2):(2*samples),:]
Y_train = [1]*samples
Y_valid = [1]*(samples/2)
Y_test  = [1]*(samples/2)
for i in range(1,10):
	file = np.loadtxt("data/mnist_digit_" + str(i) + ".csv")
	X_train = np.concatenate((X_train, file[0:samples,:]))
	X_valid = np.concatenate((X_valid, file[samples:(3*samples/2),:]))
	X_test  = np.concatenate((X_test, file[(3*samples/2):(2*samples),:]))
	Y_train += [i]*samples
	Y_valid += [i]*(samples/2)
	Y_test  += [i]*(samples/2)
		

# Normalization
X_train = 2.0*X_train/255 - 1
X_valid = 2.0*X_valid/255 - 1
X_test = 2.0*X_test/255 - 1

print X_train.shape

N = samples*len(digits)

# Initialize NN/loss
net = nn.NN(784, hdim, 10, nlayers)

# Train NN
max_epochs = 5e2
lamb = 0.01
prev_loss = 10e5
curr_loss = prev_loss-1
epoch = 0
train_loss = []
valid_loss = []
test_loss = []
while abs(curr_loss - prev_loss) > 1e-5 and epoch < max_epochs and lamb > 1e-5:
	# SGD
	perm_order = np.random.permutation(N)
	for i in perm_order:
		net.update(X_train[i,:], Y_train[i], lamb)
	# Loss
	prev_loss = curr_loss
	curr_loss = net.updateLoss(X_valid, Y_valid)
	#print "Training loss:", net.updateLoss(X_train, Y_train) , "Validation loss: ",str(curr_loss), "Learning rate : ",str(lamb), "Epoch :",str(epoch)
	if curr_loss > prev_loss:
		lamb = lamb*0.9
		#if (curr_loss-prev_loss)/prev_loss > 1e-2:
		#	break
	epoch += 1
	# Keep losses
	train_loss.append(net.updateLoss(X_train, Y_train))
	valid_loss.append(net.updateLoss(X_valid, Y_valid))
	test_loss.append(net.updateLoss(X_test, Y_test))

# Test loss
#print "Test loss: " + str(net.updateLoss(X_test, Y_test))
acc = 0.0
Nt = len(X_test)
for i in range(Nt):
	predprob = net.predictprob(X_test[i,:])
	pred = np.argmax(predprob)
	if pred == Y_test[i]:
		acc += 1
#print "Test accuracy: " + str(acc / Nt)
print "Num. layers:", nlayers, "Hid. dim:", hdim, "Epochs: ", str(epoch), "Test loss:", str(net.updateLoss(X_test, Y_test)), "Test accuracy: ", str(acc/Nt)

# Plot loss
pl.figure()
pl.plot(range(epoch), train_loss, label = "Train")
pl.plot(range(epoch), valid_loss, label = "Valid", ls = 'dashed')
pl.plot(range(epoch), test_loss, label = "Test", ls = 'dashed')
pl.xlabel("Epochs")
pl.ylabel("Loss")
pl.legend()
pl.savefig('hw3_1-5' +'_' + str(nlayers) + '_' + str(hdim) + '_loss.pdf')
