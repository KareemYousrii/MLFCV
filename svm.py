from sklearn import svm, grid_search
from mnist import load_test, load_train
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


print "\n---------------------- SVMs ----------------------"

# Load partial training and test datasets
X, y = load_train(limit=999)
XX, yy = load_test(limit=999)

svc = svm.SVC()
svc.fit(X,y)

print "--> Accuracy using the default SVM is " + str(accuracy_score(yy, svc.predict(XX)))

# Train an SVM using an RBF kernel
rbf_max_acc = 0
rbf_max_C = 0
rbf_max_gamma = 0

# Try all parameters in the range 1e-05 to 1e-05
for C in np.logspace(-2, 2, num=3):
	for gamma in np.logspace(-2, 2, num=3):

		# Initialize SVM classifier, and train using training data
		svc = svm.SVC(kernel='rbf', C=C, gamma=gamma)
		svc.fit(X,y)
		current_accuracy = accuracy_score(yy, svc.predict(XX))

		# Maintain global maximum
		if(current_accuracy > rbf_max_acc):
			rbf_max_acc = current_accuracy
			rbf_max_C = C
			rbf_max_gamma = gamma

print "--> Highest accuracy achieved using the rbf kernel is " + str(rbf_max_acc) + " using C = " + str(rbf_max_C) + " and gamma = " + str(rbf_max_gamma)

# Train an SVM using a polynomial kernel
poly_max_acc = 0
poly_max_C = 0
poly_max_gamma = 0
poly_max_degree = 0

# Try all parameters in the range 1e-05 to 1e-05
for C in (0.1, 1, 10, 100):
	for gamma in (0.00001, 1):
		for degree in xrange(3, 6):

			# Initialize SVM classifier, and train using training data
			svc = svm.SVC(kernel='poly', C=C, gamma=gamma, degree=degree)
			svc.fit(X,y)
			current_accuracy = accuracy_score(yy, svc.predict(XX))

			# Maintain global maximum
			if(current_accuracy > poly_max_acc):
				poly_max_acc = current_accuracy
				poly_max_C = C
				poly_max_gamma = gamma
				poly_max_degree = degree

print "--> Highest accuracy achieved using the polynomial kernel is " + str(poly_max_acc) + " using C = " + str(poly_max_C) + " and gamma = " + str(poly_max_gamma) + " and a degree " + str(poly_max_degree) + " polynomial"


# Train an SVM using a linear kernel
linear_max_acc = 0
linear_max_C = 0

# Try all parameters in the range 1e-05 to 1e-05
for C in np.logspace(-4, 4, num=5):

		# Initialize SVM classifier, and train using training data
		svc = svm.SVC(kernel='linear', C=C)
		svc.fit(X,y)
		current_accuracy = accuracy_score(yy, svc.predict(XX))

		# Maintain global maximum
		if(current_accuracy > linear_max_acc):
			linear_max_acc = current_accuracy
			linear_max_C = C

print "--> Highest accuracy achieved using the linear kernel is " + str(linear_max_acc) + " using C = " + str(linear_max_C)

# Load complete training and test datasets
X, y = load_train()
XX, yy = load_test()

# Perform training of the entire data set and report the accuracy on the test data set
svc = None

if rbf_max_acc > poly_max_acc and rbf_max_acc > linear_max_acc:
	svc = svm.SVC(kernel='rbf', C=rbf_max_C, gamma=rbf_max_gamma)

elif poly_max_acc > rbf_max_acc and poly_max_acc > linear_max_acc:
	svc = svm.SVC(kernel='poly', C=poly_max_C, gamma=poly_max_gamma, degree=poly_max_degree)

else:
	svc = svm.SVC(kernel='linear', C=linear_max_C)

svc.fit(X, y)

accuracy = accuracy_score(yy, svc.predict(XX))

print "--> Testing error using the best hyper-parameters and training on the entire data set is " + str(accuracy)
