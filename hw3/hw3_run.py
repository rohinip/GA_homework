############################
## Rohini Pandhi          ##          
## Homework Assignment 3  ##
## Run File               ##
## 02/06/2014             ##
############################

from hw3 import load_heart_data, plot_histogram, random_forest

(fulldata, features, outcomes) = load_heart_data()
# features = age, sex, chest_pain_type, rest_blood_pressure, serum_chol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercised_angina, old peak, slope_of_peak, vessels_colored, thal
# outcomes = 1 if no disease, 2 if disease 

plot_histogram(fulldata)

#random_forest(features, outcomes, 10)

print 
print "====== DONE ======"
