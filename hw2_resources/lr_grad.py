import numpy as np
import math

def NLL(X, Y, w, w0):
    return np.mean(np.log(1 + np.exp(-Y * (w0 + np.dot(X,w).reshape(len(X),1)))))

def logisticLoss(X, Y, w, w0, lamb):
    return NLL(X, Y, w, w0) + lamb * np.linalg.norm(w)

def logisticGrad(X, Y, w, w0, lamb):
    dEdw0 = -np.sum(Y * np.exp(-Y * (w0 + np.dot(X,w).reshape(len(X),1))) / (1 + np.exp(-Y * (w0 + np.dot(X,w).reshape(len(X),1)))))
    dEdw = -np.sum(Y*X*np.exp(Y*((w0) + np.dot(X,w).reshape(len(X),1)))/(1+np.exp(Y*((w0) + np.dot(X,w).reshape(len(X),1))))) + 2*lamb*w
    return dEdw, dEdw0

def GradientDescent(X, Y, loss, grad, lamb, init, epochs, step, eps):
    w = init[:-1]
    w0 = init[-1]
    if epochs:
        t = 0
        prev_loss = 1e8
        curr_loss = prev_loss-1
        while t < epochs and abs(curr_loss - prev_loss) > eps:
            t += 1
            gradw, gradw0 = grad(X, Y, w, w0, lamb)
            w = w - step * gradw
            w0 = w0 -  step * gradw0
            prev_loss = curr_loss
            curr_loss = loss(X, Y, w, w0, lamb)
    else:
        prev_loss = 1e8
        curr_loss = prev_loss-1
        while abs(curr_loss - prev_loss) > eps:
            gradw, gradw0 = grad(X, Y, w, w0, lamb)
            w += step * gradw
            w0 += step * gradw0
            prev_loss = curr_loss
            curr_loss = loss(X, Y, w, w0, lamb)
    return w, w0
