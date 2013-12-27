############################
## Rohini Pandhi          ##          
## Homework Assignment 1  ##
## 01/09/2014             ##
############################

# Instructions: 
# 3. Perform cross n-fold validation
# 3a. Call (3) while parameterizing n, showing the accuracy result from each test
# 4a. Provide description of the problem you are solving
# 4b. Provide description of problems that might arise when approaching that problem
# 4c. Provide your results and future direction
# 5. Push to github, provide link on GDocs

import argparse
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

iris = datasets.load_iris()
features_training = iris.data
species_training = iris.target
knn_neighbors = 3
validation_folds = 3

def knn(n, x_training, y_training):
  neigh = KNeighborsClassifier(n) #number of neighbors to use
  clf = neigh.fit(x_training, y_training) #fit the KNN model using training info
  return clf

def naive_bayes(x_training, y_training):
  nbayes = GaussianNB()
  clf = nbayes.fit(x_training, y_training) #fit the NB model using training info
  return clf

def cross_validate(folds, x, y, model):
  fold = KFold(len(y), n_folds=folds, shuffle=True)

parser = argparse.ArgumentParser(description='Select KNN or Naive Bayes Classifier')
parser.add_argument('-c', '--classifier', help='a classifier type: KNN or NB (if none selected, default is Naive Bayes', required=True)
args = parser.parse_args()

if (args.classifier.upper()=='KNN'):
  model = knn(knn_neighbors, features_training, species_training)
else:
  model = naive_bayes(features_training, species_training)

cross_validate(validation_folds, features_training, species_training, model)

######## Appendix ########
## http://en.wikipedia.org/wiki/Iris_flower_data_set
## http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
## http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
## http://scikit-learn.org/stable/modules/cross_validation.html#k-fold 
## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html
########################## 
