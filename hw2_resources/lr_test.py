from numpy import *
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return model.predict(x)

# parameters
name = '3'
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
lambs = [0.1, 1, 10, 100, 250, 350, 500, 1000]
penalties = ['l1', 'l2']
accs = {}
w1s = {}
w2s = {}
w_norms = {}
accs_valid = {}
w1s_valid = {}
w2s_valid = {}
w_norms_valid = {}

i = 0
for penalty in penalties:
	accs[penalty] = []
	w1s[penalty] = []
	w2s[penalty] = []
	w_norms[penalty] = []
	accs_valid[penalty] = []
	for lamb in lambs:
		model = LogisticRegression(penalty = penalty, C = 1.0/lamb).fit(X, Y)

		# classification error
		accs[penalty].append(model.score(X,Y.reshape(len(Y),)))
		accs_valid[penalty].append(model.score(X_valid,Y_valid.reshape(len(Y_valid),)))

		# weights
		w1s[penalty].append(model.coef_[0][0])
		w2s[penalty].append(model.coef_[0][1])
		w_norms[penalty].append(linalg.norm(model.coef_))

		# plot training results
		if lamb in [0.1, 1, 10, 100]:
			plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'Lambda = ' + str(lamb))
			#pl.savefig('hw2_1-2_' + str(4 + i) + '.pdf')
			i += 1

# Plot accuracy
fig = plt.figure()
for penalty in penalties:
	col = 'r' if penalty == 'l1' else 'b'
	#plt.plot(lambs, accs[penalty], label = penalty + '-train', color = col)
	#plt.plot(lambs, accs_valid[penalty], ls = 'dashed', label = penalty + '-valid', color = col)

axes = plt.gca()
axes.set_ylim([0.45, 1.05])
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.legend()
#plt.savefig('hw2_1-2_1.pdf')

# Plot weight norms
fig = plt.figure()
for penalty in penalties:
	col = 'r' if penalty == 'l1' else 'b'
	plt.plot(lambs, w_norms[penalty], label = penalty, color = col)

axes = plt.gca()
axes.set_ylim([0, 4])
plt.xlabel('lambda')
plt.ylabel('|w|')
plt.legend()
#plt.savefig('hw2_1-2_2.pdf')

# Plot weights
fig = plt.figure()
for penalty in penalties:
	col = 'r' if penalty == 'l1' else 'b'
	plt.plot(lambs, w1s[penalty], label = 'w1, ' + penalty, color = col)
	plt.plot(lambs, w2s[penalty], label = 'w2, ' + penalty, ls = 'dashed', color = col)

axes = plt.gca()
axes.set_ylim([-1, 4])
plt.xlabel('lambda')
plt.ylabel('w_i')
plt.legend()
#plt.savefig('hw2_1-2_3.pdf')

# Plot decision boundaries
plotDecisionBoundary(X_valid, Y_valid, predictLR, [0.5], title = 'LR Validate')
plotDecisionBoundary(X_test, Y_test, predictLR, [0.5], title = 'LR Test')
pl.show()
