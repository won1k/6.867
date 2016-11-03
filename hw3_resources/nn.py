import numpy as np
import itertools

class softmax(object):
    def __init__(self, dim, odim):
        self.odim   = odim    # output dim
        self.dim    = dim       # input dim       
        self.Wo     = np.zeros((self.odim,))
        self.W      = np.random.randn(self.dim, self.odim) / np.sqrt(self.dim)
        self.prob   = np.ones((self.odim,)) / float(self.odim)
        
    def forward(self, x):
        z           = self.Wo + np.dot(x, self.W)
        self.prob   = np.exp(z - np.max(z))
        self.prob  /= np.sum(self.prob)
        return self.prob

    # Update bias, accumulate wordvec gradient, return dlogP[y]/dx
    def backprop(self, x, lrate, y):
        grad       = -self.prob
        grad[y]   += 1.0  # dlogP[y]/dz
        xdelta     = np.dot(self.W, grad)  # dlogP[y]/dx
        self.Wo   += lrate * grad
        self.W    += lrate * np.outer(x, grad)
        return xdelta


class hidden(object):
    def __init__(self, idim, hdim):
        self.idim = idim
        self.hdim = hdim
        self.W    = np.random.randn(self.idim, self.hdim) / np.sqrt(self.idim)
        self.Wo   = np.zeros(self.hdim,)
        self.f    = np.zeros((self.hdim,))  # activation of output units

    def forward(self, x):
    	np.maximum(self.Wo + np.dot(x, self.W), 0, self.f)
        return self.f 

    def backprop(self, x, lrate, delta):
        grad       = (self.f > 0) * delta  # dJ/dz = df/dz * delta 
        xdelta     = np.dot(self.W, grad)         # dJ/dx to be returned
        self.Wo   += lrate * grad
        self.W    += lrate * np.outer(x, grad)
        return xdelta

class NN(object):
	def __init__(self, dim, hdim, odim, nlayers):
		self.dim     = dim       # input dimension
		self.nlayers = nlayers	 # number of layers
		self.hdim    = hdim      # number of hidden layer units
		self.odim    = odim      # output dimension
		self.loss    = 0         # loss

		# Initialize network
		self.hiddenL = {}
		for i in range(self.nlayers):
			if i == 0:
				self.hiddenL[i] = hidden(self.dim, self.hdim)
			else:
				self.hiddenL[i] = hidden(self.hdim, self.hdim)
		self.outputL = softmax(self.hdim, self.odim)

	def prob(self, x, y):
		for i in range(self.nlayers):
			x = self.hiddenL[i].forward(x)
		prob = self.outputL.forward(x)
		return prob[y]

	def update(self, x, y, lrate):
		z = {}
		# Forward pass
		z[0] = self.hiddenL[0].forward(x)
		for i in range(1, self.nlayers):
			z[i] = self.hiddenL[i].forward(z[i-1])
		prob = self.outputL.forward(z[self.nlayers-1])

		# Backward pass (Backprop)
		dh = self.outputL.backprop(z[self.nlayers-1], lrate, y)
		for i in range(self.nlayers, 1, -1):
			dh = self.hiddenL[i-1].backprop(z[i-2], lrate, dh)
		dx = self.hiddenL[0].backprop(x, lrate, dh)

		return prob[y]

	def updateLoss(self, X, Y):
		self.loss = 0.0
		for x, y in itertools.izip(X, Y):
			self.loss -= np.log(self.prob(x, y))
		return self.loss

    




