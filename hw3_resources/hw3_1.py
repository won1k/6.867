import numpy as np
import nn

# Load data
data = np.genfromtxt('data/data_3class.csv', delimiter = ' ')
X = data[:,0:2]
Y = [int(y) for y in data[:,-1]]
n = len(X)

# Initialize NN/loss
net = nn.NN(2, 5, 3, 1)

# Train NN
epochs = 1000
lamb = 0.005
for epoch in range(epochs):
	# SGD
	perm_order = np.random.permutation(n)
	for i in perm_order:
		net.update(X[i,:], Y[i], lamb)
	# Loss
	print "Training loss: " + str(net.updateLoss(X,Y))