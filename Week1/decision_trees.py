from mnist import load_train, load_test
from sklearn import tree, cross_validation
from sklearn.externals.six import StringIO 
import numpy as np
import pydot 
import matplotlib.pyplot as plt

print "\n---------------------- DecisionTrees ----------------------"

# Load partial training and test datasets
X, y = load_train(limit=999)
XX, yy = load_test(limit=999)

# Initialize a decision tree classifier and fit it on the first 1000 training samples
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)

print "An accuracy of " + str(clf.score(XX,yy)) + " was achieved on the test data set using the default DecisionTreeClassifer parameters"

# Determine the best parameters for our problem
max_accuracy = 0
best_criterion = None
best_max_depth = 0
best_max_features = 0

for criterion in ('gini', 'entropy'):
	for max_depth in np.arange(4, 804, 20):
		for max_features in ('sqrt', 'log2'):
			clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features)
			clf.fit(X,y)
			current_accuracy = clf.score(XX,yy)

			if(current_accuracy > max_accuracy):
				max_accuracy = current_accuracy
				best_criterion = criterion
				best_max_depth = max_depth
				best_max_features = max_features

print "A maximum accuracy of " + str(max_accuracy) + " was achieved using max_depth = " + str(best_max_depth) + " max_features = " + str(best_max_features) + " and the " + best_criterion + " criterion"

# Load partial training and test datasets
X, y = load_train()
XX, yy = load_test()

# Train a DecisionTreeClassifier on the entire training data set using the best parameters
clf = tree.DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth, max_features=best_max_features)
clf.fit(X,y)

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("BestDecisionTree.pdf")

scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print "The mean score achieved using cross validation on a decision_tree is " + str(scores.mean())

importances = clf.feature_importances_
importances = importances.reshape(28, 28)
plt.matshow(importances, cmap=plt.cm.hot)
plt.show()

