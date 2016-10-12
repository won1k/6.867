import numpy as np
from lr_grad import *
import matplotlib.pyplot as plt

train = np.loadtxt('data/data1_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Perform gradient descent for variable iterations
step = 1e-3
init = np.array([1,1,0])
lamb = 0
epochs = [10, 50, 100, 1000, 5000, 10000]
eps = 1e-8

w_norms = []
for epoch in epochs:
    print epoch
    w, w0 = GradientDescent(X, Y, logisticLoss, logisticGrad, lamb, init, epoch, step, eps)
    w_norms.append(np.linalg.norm(w))

plt.plot(epochs, w_norms)

# With regularization
lamb = 1
w_norms = []
for epoch in epochs:
    print epoch
    w, w0 = GradientDescent(X, Y, logisticLoss, logisticGrad, lamb, init, epoch, step, eps)
    w_norms.append(np.linalg.norm(w))

plt.plot(epochs, w_norms)

plt.savefig("hw2_1-1_1.pdf")
