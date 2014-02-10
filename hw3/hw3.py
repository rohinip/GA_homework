############################
## Rohini Pandhi          ##          
## Homework Assignment 3  ##
## 02/06/2014             ##
############################

import json
import os
import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# Dataset from http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
def load_heart_data():
  data = open('/home/GA8/GA_homework/hw3/heart.dat').read()
  data = data.split('\n') #split rows
  features = []
  outcomes = []
  for d in data: 
    d = d.split()
    feature_row = d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12]
    features.append(feature_row)
    outcomes.append(d[13])
  features = np.array(features)
  outcomes = np.array(outcomes)
  return (features, outcomes)

def plot_histogram(features):
  num_bins = 20 #20 bars in a single histogram class
  
  # Attributes
  age = features[0::,0].astype(np.float)
  sex = features[0::,1].astype(np.float)
  chest_pain_type = features[0::,2].astype(np.float)
  resting_blood_pressure = features[0::,3].astype(np.float)
  serum = features[0::,4].astype(np.float)
  fasting_blood_sugar = features[0::,5].astype(np.float)
  resting_ecg = features[0::,6].astype(np.float)
  max_heart_rate = features[0::,7].astype(np.float)
  exercise_angina = features[0::,8].astype(np.float)
  oldpeak = features[0::,9].astype(np.float)
  slope_peak = features[0::,10].astype(np.float)
  colored_vessels = features[0::,11].astype(np.float)

  # Mean and Standard Deviation Calculations
  mean_age = np.mean(age)
  mean_rbp = np.mean(resting_blood_pressure)
  std_dev_age = np.std(age)
  std_dev_rbp = np.std(resting_blood_pressure)

  # T statistic and P value between attributes
  t_statistic, p_value = ttest_ind(age, resting_blood_pressure)

  # Plot Histogram of Age Attribute
  plt.figure(1)
  plt.hist(age[0:], 
           bins=num_bins,
           range=(0,100),
           #histtype='barstacked',
           facecolor='red', #green
           label='Class',
           alpha=.5)
  plt.xlabel('Age')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_all_patients.png')

# Random Forest Classifier
def random_forest(x_training, y_training, cv):
  clf = RandomForestClassifier(n_estimators=1000)
  score = cross_val_score(clf, x_training, y_training, cv=cv)
  print "%s-fold cross validation accuracy: %s" % (cv,sum(score)/score.shape[0])
  clf = clf.fit(x_training, y_training)
  preds = clf.predict(x_training)
  #print "predictions counter = %s" % (Counter(clf.predict(x_training))
  fp = 0
  tp = 0
  fn = 0
  tn = 0
  for a in range(len(y_training)):
    if y_training[a]==preds[a]:
      if preds[a]==0: #might have to make this 1 bc of heart data
        tn+=1
      elif preds[a]==1:
        tp+=1
    elif preds[a]==1: fp+=1
    elif preds[a]==0: fn+=1

  print 'correct positives:', tp
  print 'correct negatives:', tn
  print 'false positives:', fp
  print 'false negatives:', fn
  precision = float(tp)/(tp+fp)
  recall = float(tp)/(tp+fn)
  fpr = float(fp)/(fp+tn)
  fdr = float(fp)/(fp+tp)
  prediction_accuracy = (100*float(tp+tn)/(tp+tn+fp+fn),'%') 
  print 'precision:', precision
  print 'recall:',recall
  print 'fpr:', fpr
  print 'fdr:', fdr
  print 'prediction accuracy:', prediction_accuracy

  return clf
