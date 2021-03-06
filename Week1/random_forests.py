from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from mnist import load_train, load_test
import numpy as np
import matplotlib.pyplot as plt


print "\n---------------------- RandomForests ----------------------"

# Load partial training and test datasets
X, y = load_train(limit=999)
XX, yy = load_test(limit=999)

# Initialize a RandomForestClassifier with default hyper-parameters
# and fit it on the training data set
clf = RandomForestClassifier()
clf.fit(X,y)

print "An accuracy of " + str(clf.score(XX,yy)) + " was achieved on the test data set using the default RandomForestClassifier parameters"

# Determine the best parameters for our problem
max_accuracy = 0
best_criterion = None
best_max_depth = 0
best_max_features = 0
best_n_estimators = 0

for criterion in ('gini', 'entropy'):
	for max_depth in np.arange(4, 804, 20):
		for max_features in ('sqrt', 'log2'):
			for n_estimators in (10, 30, 60):
				clf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators, n_jobs=-1)
				clf.fit(X,y)
				current_accuracy = clf.score(XX,yy)

				if(current_accuracy > max_accuracy):
					max_accuracy = current_accuracy
					best_criterion = criterion
					best_max_depth = max_depth
					best_max_features = max_features
					best_n_estimators = n_estimators

print "A maximum accuracy of " + str(max_accuracy) + " was achieved using max_depth = " + str(best_max_depth) + ", max_features = " + str(best_max_features) + ", " +  str(best_n_estimators) + " estimators, and the " + best_criterion + " criterion"

# Load complete training and test datasets
X, y = load_train()
XX, yy = load_test()

# Train a DecisionTreeClassifier on the entire training data set using the best parameters
clf = RandomForestClassifier(criterion=best_criterion, max_depth=best_max_depth, max_features=best_max_features, n_estimators=best_n_estimators, n_jobs=-1)
clf.fit(X,y)

scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print "The mean score achieved using cross validation on a random_forest is " + str(scores.mean())

# Visualize the pixel importances
importances = clf.feature_importances_
importances = importances.reshape(28, 28)
plt.matshow(importances, cmap=plt.cm.hot)
plt.show()