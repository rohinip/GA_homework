############################
## Rohini Pandhi          ##          
## Homework Assignment 1  ##
## 01/09/2014             ##
############################


# Imports
import argparse
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

# Load Iris Dataset
def load_iris_data(): 
  iris = datasets.load_iris()
  return (iris.data, iris.target, iris.target_names) #features, species, species_options  

# K Nearest Neighbors
#   other useful methods of this knn object:
#     fit(X_train, y_train) --> fit the model using a training set
#     predict(X_classify) --> to predict a result using the trained model
#     score(X_test, y_test) --> to score the model using a test set
def knn(x_training, y_training, n=3):
  neigh = KNeighborsClassifier(n) #number of neighbors to use
  clf = neigh.fit(x_training, y_training) #fit the KNN model using training info
  return clf

# Naive Bayes
#  other useful methods of this knn object:
#    fit(X_train, y_train) --> fit the model using a training set
#    predict(X_classify) --> to predict a result using the trained model
#    score(X_test, y_test) --> to score the model using a test set
def nb(x_training, y_training):
  nbayes = GaussianNB()
  clf = nbayes.fit(x_training, y_training) #fit the NB model using training info
  return clf

# Cross Validation
def cross_validate(features, classifications, clf, number_folds):
  
  #KFold provides train/test indices (shuffle enabled to create random indices)
  kfold_indices = KFold(len(classifications), n_folds=number_folds, shuffle=True)
  
  #for each training and testing slices, run the classifier and score the results
  k_score_total = 0
  for train_slice, test_slice in kfold_indices: #will loop through number_folds times
    model = clf(features[[train_slice]], classifications[[train_slice]])
    k_score = model.score(features[[test_slice]], classifications[[test_slice]])
    k_score_total += k_score

  #return average (mean) accuracy
  return (k_score_total/number_folds)


######## Appendix ########
## http://en.wikipedia.org/wiki/Iris_flower_data_set
## http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
## http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
## http://scikit-learn.org/stable/modules/cross_validation.html#k-fold 
## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html
########################## 
