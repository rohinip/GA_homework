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
from scipy import stats
from scipy.stats import ttest_ind, ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# Dataset from http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
def load_heart_data():
  data = open('/home/GA8/GA_homework/hw3/heart.dat').read()
  data = data.split('\n') #split rows
  fulldata = []
  features = []
  outcomes = []
  for d in data: 
    d = d.split()
    fulldata_row = d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]
    feature_row = d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12]
    fulldata.append(fulldata_row)
    features.append(feature_row)
    outcomes.append(d[13])
  fulldata = np.array(fulldata)
  features = np.array(features)
  outcomes = np.array(outcomes)
  return (fulldata, features, outcomes)

def plot_histogram(fulldata):
  num_bins = 20 #20 bars in a single histogram class
  
  # Attributes
  age = fulldata[0::,0].astype(np.float)
  sex = fulldata[0::,1].astype(np.float)
  chest_pain_type = fulldata[0::,2].astype(np.float)
  resting_blood_pressure = fulldata[0::,3].astype(np.float)
  serum = fulldata[0::,4].astype(np.float)
  fasting_blood_sugar = fulldata[0::,5].astype(np.float)
  resting_ecg = fulldata[0::,6].astype(np.float)
  max_heart_rate = fulldata[0::,7].astype(np.float)
  exercise_angina = fulldata[0::,8].astype(np.float)
  oldpeak = fulldata[0::,9].astype(np.float)
  slope_peak = fulldata[0::,10].astype(np.float)
  colored_vessels = fulldata[0::,11].astype(np.float)

  no_disease_age = []
  no_disease_sex =[]
  no_disease_chest_pain_type =[]
  no_disease_resting_blood_pressure =[]
  no_disease_serum =[]
  no_disease_fasting_blood_sugar =[]
  no_disease_resting_ecg =[]
  no_disease_max_heart_rate =[]
  no_disease_exercise_angina =[]
  no_disease_oldpeak =[]
  no_disease_slope_peak =[]
  no_disease_colored_vessels =[]

  disease_age = []
  disease_sex = []
  disease_chest_pain_type = []
  disease_resting_blood_pressure = []
  disease_serum = []
  disease_fasting_blood_sugar = []
  disease_resting_ecg = []
  disease_max_heart_rate = []
  disease_exercise_angina = []
  disease_oldpeak = []
  disease_slope_peak = []
  disease_colored_vessels = []

  for n in fulldata: 
    if n[13] == '1': 
      no_disease_age.append(n[0])
      no_disease_sex.append(n[1])
      no_disease_chest_pain_type.append(n[2])
      no_disease_resting_blood_pressure.append(n[3])
      no_disease_serum.append(n[4])
      no_disease_fasting_blood_sugar.append(n[5])
      no_disease_resting_ecg.append(n[6])
      no_disease_max_heart_rate.append(n[7])
      no_disease_exercise_angina.append(n[8])
      no_disease_oldpeak.append(n[9])
      no_disease_slope_peak.append(n[10])
      no_disease_colored_vessels.append(n[11])
    elif n[13] == '2': 
      disease_age.append(n[0])
      disease_sex.append(n[1])
      disease_chest_pain_type.append(n[2])
      disease_resting_blood_pressure.append(n[3])
      disease_serum.append(n[4])
      disease_fasting_blood_sugar.append(n[5])
      disease_resting_ecg.append(n[6])
      disease_max_heart_rate.append(n[7])
      disease_exercise_angina.append(n[8])
      disease_oldpeak.append(n[9])
      disease_slope_peak.append(n[10])
      disease_colored_vessels.append(n[11])

  no_disease_age = np.array(no_disease_age).astype(np.float)
  no_disease_sex = np.array(no_disease_sex).astype(np.float)
  no_disease_chest_pain_type = np.array(no_disease_chest_pain_type).astype(np.float)
  no_disease_resting_blood_pressure = np.array(no_disease_resting_blood_pressure).astype(np.float)
  no_disease_serum = np.array(no_disease_serum).astype(np.float)
  no_disease_fasting_blood_sugar = np.array(no_disease_fasting_blood_sugar).astype(np.float)
  no_disease_resting_ecg = np.array(no_disease_resting_ecg).astype(np.float)
  no_disease_max_heart_rate = np.array(no_disease_max_heart_rate).astype(np.float)
  no_disease_exercise_angina = np.array(no_disease_exercise_angina).astype(np.float)
  no_disease_oldpeak = np.array(no_disease_oldpeak).astype(np.float)
  no_disease_slope_peak = np.array(no_disease_slope_peak).astype(np.float)
  no_disease_colored_vessels = np.array(no_disease_colored_vessels).astype(np.float)

  disease_age = np.array(disease_age).astype(np.float)
  disease_sex = np.array(disease_sex).astype(np.float)
  disease_chest_pain_type = np.array(disease_chest_pain_type).astype(np.float)
  disease_resting_blood_pressure = np.array(disease_resting_blood_pressure).astype(np.float)
  disease_serum = np.array(disease_serum).astype(np.float)
  disease_fasting_blood_sugar = np.array(disease_fasting_blood_sugar).astype(np.float)
  disease_resting_ecg = np.array(disease_resting_ecg).astype(np.float)
  disease_max_heart_rate = np.array(disease_max_heart_rate).astype(np.float)
  disease_exercise_angina = np.array(disease_exercise_angina).astype(np.float)
  disease_oldpeak = np.array(disease_oldpeak).astype(np.float)
  disease_slope_peak = np.array(disease_slope_peak).astype(np.float)
  disease_colored_vessels = np.array(disease_colored_vessels).astype(np.float)

  # Plot Histogram of Age Attribute
  plt.figure(1)
  plt.hist(no_disease_age[0:], 
           bins=num_bins,
           range=(0,100),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_age[0:],
           bins=num_bins, 
           range=(0,100),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Age')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_age_patients.png')

  # Plot Histogram of Resting Blood Pressure Attribute
  plt.figure(2)
  plt.hist(no_disease_resting_blood_pressure[0:],
           bins=num_bins,
           range=(90,200),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_resting_blood_pressure[0:],
           bins=num_bins,
           range=(90,200),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Resting Blood Pressure')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_resting_blood_pressure_patients.png')

  # Plot Histogram of Serum Attribute
  plt.figure(3)
  plt.hist(no_disease_serum[0:],
           bins=num_bins,
           range=(100,600),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_serum[0:],
           bins=num_bins,
           range=(100,600),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Serum')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_serum_patients.png')

  # Plot Histogram of Max Heart Rate Attribute
  plt.figure(4)
  plt.hist(no_disease_max_heart_rate[0:],
           bins=num_bins,
           range=(90,210),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_max_heart_rate[0:],
           bins=num_bins,
           range=(90,210),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Max Heart Rate')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_max_heart_rate_patients.png')

  # Plot Histogram of Oldpeak Attribute
  plt.figure(5)
  plt.hist(no_disease_oldpeak[0:],
           bins=num_bins,
           range=(0,4),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_oldpeak[0:],
           bins=num_bins,
           range=(0,4),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Oldpeak')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_oldpeak_patients.png')

  # Plot of Slope Peak
  plt.figure(6)
  plt.hist(no_disease_slope_peak[0:],
           bins=num_bins,
           range=(0,4),
           facecolor='red',
           label='Class',
           alpha=.5)
  plt.hist(disease_slope_peak[0:],
           bins=num_bins,
           range=(0,4),
           facecolor='blue',
           label='Class',
           alpha=.5)
  plt.xlabel('Slope Peak')
  plt.ylabel('Number of Patients')
  plt.savefig('hist_slope_peak_patients.png')

# Random Forest Classifier
def random_forest(x_training, y_training, cv):
  clf = RandomForestClassifier(n_estimators=100)

  ### Need to slice the data set into training and test data ###
  score = cross_val_score(clf, x_training, y_training, cv=cv)
  print "%s-fold cross validation accuracy: %s" % (cv,sum(score)/score.shape[0])
  ### ### ### ### ###

  clf = clf.fit(x_training, y_training)
  preds = clf.predict(x_training)
  #print "predictions counter = %s" % (Counter(clf.predict(x_training))
  fp = 0
  tp = 0
  fn = 0
  tn = 0
  for a in range(len(y_training)):
    if y_training[a]==preds[a]:
      if preds[a]=="2": #no disease
        tn+=1
      elif preds[a]=="1": #disease
        tp+=1
    elif preds[a]=="2": fp+=1
    elif preds[a]=="1": fn+=1

  print 'correct positives:', tp
  print 'correct negatives:', tn
  print 'false positives:', fp
  print 'false negatives:', fn
#  precision = float(tp)/(tp+fp)
#  recall = float(tp)/(tp+fn)
#  fpr = float(fp)/(fp+tn)
#  fdr = float(fp)/(fp+tp)
#  prediction_accuracy = (100*float(tp+tn)/(tp+tn+fp+fn),'%') 
#  print 'precision:', precision
#  print 'recall:',recall
#  print 'fpr:', fpr
#  print 'fdr:', fdr
#  print 'prediction accuracy:', prediction_accuracy

  return clf
