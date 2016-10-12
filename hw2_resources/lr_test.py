from numpy import *
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
model = LogisticRegression().fit(X, Y)

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    model.predict(x)

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show()