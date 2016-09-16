'''
Homework 1, 6.867
Won Lee
'''

import numpy as np
import pylab as pl

# Problem 1

def getData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean,gaussCov,quadBowlA,quadBowlb) 

# 1.1

def gradientDescent(obj, grad, init, step, eps):
	prev_loss = 1e9
	curr_loss = 0
	x = init
	while abs(curr_loss - prev_loss) >= eps:
		x -= step * grad(x)
		prev_loss = curr_loss
		curr_loss = obj(x)
		print "Current loss: " + str(curr_loss)
	return x



def negGauss()

def finiteDiff(obj, diff_size, x):

def 