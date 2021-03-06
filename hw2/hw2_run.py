############################
## Rohini Pandhi          ##          
## Homework Assignment 2  ##
## Run File               ##
## 01/23/2014             ##
############################

import argparse
from hw2 import load_iris_data, cross_validate, knn, nb, lr

# Load Iris dataset
(features, species, species_names) = load_iris_data()

# Argument parse
parser = argparse.ArgumentParser(description='Select KNN, Naive Bayes, or Logistic Regression')
parser.add_argument('-c', '--classifier', help='a classifier type: KNN, NB, or LR (if none selected, default will run all', required=False)
args = parser.parse_args()

try: 
  if (args.classifier.upper() == "KNN"): 
    classifier_list = [("KNN",knn)]
  elif (args.classifier.upper() == "NB"):
    classifier_list = [("Naive Bayes",nb)]
  elif (args.classifier.upper() == "LR"): 
    classifier_list = [("Logistic Regression",lr)]
except:
  classifier_list = [("KNN",knn), ("Naive Bayes",nb), ("Logistic Regression",lr)] #using imported functions above

# Loop through each tuple of the classifier list
for (classifier_string, classifier_function) in classifier_list:
  
  print "\n-------- %s --------" % classifier_string

  best_kfolds = 0
  best_cv = 0

  # Define specific set of folds that split the dataset in an even way
  for kfolds in [2,3,5,10,15,30,50,75]: 

    cv = cross_validate(features, species, classifier_function, kfolds)

    if cv > best_cv:
      best_cv = cv
      best_kfolds = kfolds
    
    # For each fold, print out it's accuracy in the cross validation function
    print "Fold <<%s>> :: Accuracy <<%s>>" % (kfolds, cv)

  # At the end, print the highest accuracy fold
  print "%s :: Highest Accuracy: <<%s>> :: Fold <<%s>>\n" % (classifier_string, best_cv, best_kfolds)
