############################
## Rohini Pandhi          ##          
## Homework Assignment 2  ##
## 01/23/2014             ##
############################

import argparse
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

def load_iris_data():
  iris = datasets.load_iris()
  return (iris.data, iris.target, iris.target_names) #features, species, species_options  

# Logistic Regression
def lr(x_training, y_training, c=1.0):
  logreg = LogisticRegression(C=c) #inverse regularization strength (smaller C => stronger regularization)
  clf = logreg.fit(x_training, y_training)
  return clf

# K Nearest Neighbors
def knn(x_training, y_training, n=3):
  neigh = KNeighborsClassifier(n) #number of neighbors to use
  clf = neigh.fit(x_training, y_training) #fit the KNN model using training info
  return clf

# Naive Bayes
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



