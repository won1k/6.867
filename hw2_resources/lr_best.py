from numpy import *
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return model.predict(x)

names = ['1','2','3','4']
w_points = [w+0.75 for w in range(4)]
x_values = [w+1 for w in range(4)]
fig = plt.figure()
plt.xticks(x_values)
best_pens = []
best_lambs = []
best_accs = []

for name in names:
	print "Dataset: " + name
	print '======Training======'
	# load data from csv files
	train = loadtxt('data/data'+name+'_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	print '======Validation======'
	# load data from csv files
	validate = loadtxt('data/data'+name+'_validate.csv')
	X_valid = validate[:,0:2]
	Y_valid = validate[:,2:3]

	print '======Testing======'
	# load data from csv files
	test = loadtxt('data/data'+name+'_test.csv')
	X_test = test[:,0:2]
	Y_test = test[:,2:3]

	# Carry out training.
	lambs = [0.01, 0.1, 0.3, 0.5, 1, 5, 10, 100, 500]
	penalties = ['l1', 'l2']
	accs = {}

	# Train for L1/2 and lambs/performance
	best_pen = None
	best_acc = 0
	best_lamb = 0
	for penalty in penalties:
		for lamb in lambs:
			model = LogisticRegression(penalty = penalty, C = 1.0/lamb).fit(X, Y)
			# classification error
			acc = model.score(X_valid, Y_valid.reshape(len(Y_valid),))
			if acc > best_acc:
				best_pen = penalty
				best_lamb = lamb
				best_acc = acc

	best_pens.append(best_pen)
	best_lambs.append(best_lamb)

	# Get accuracy on test
	model = LogisticRegression(penalty = best_pen, C = 1.0/best_lamb).fit(X, Y)
	best_accs.append(model.score(X_test, Y_test.reshape(len(Y_test),)))

print(best_pens)
print(best_lambs)
print(best_accs)

plt.bar(w_points, best_accs, width = 0.5)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.savefig('hw2_1-3_1.pdf')