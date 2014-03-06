import numpy as np
import csv as csv

# Import our data
csv_file_object = csv.reader(open('/home/GA8/titanic/train.csv', 'rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

# Calculate some summary stats
number_passengers = np.size(data[0::,0].astype(np.float)) # takes every row, first column (numpy syntax)
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print "---- Summary Stats ----"
print "Number of passengers in training set = %s" % number_passengers
print "Number of survivors in training set = %s" % number_survived
print "Survival rate of training set = %s" % proportion_survivors

# Calculate some gender-based stats
women_only_stats = data[0::,4] == "female" # rows of women passengers
men_only_stats = data[0::,4] == "male" # rows of men passengers

women_onboard_index = data[women_only_stats, 0].astype(np.float) # grab rows from data set that are women and just their PassengerID (first column)
men_onboard_index = data[men_only_stats, 0].astype(np.float)

number_women_onboard = np.size(data[women_only_stats, 0].astype(np.float))
number_men_onboard = np.size(data[men_only_stats, 0].astype(np.float))

print "---- Gender-Based Stats ----"
print "Number of women onboard = %s" % number_women_onboard
print "Number of men onboard = %s" % number_men_onboard

number_women_survivors = np.sum(data[women_only_stats, 1].astype(np.float))
number_men_survivors = np.sum(data[men_only_stats, 1].astype(np.float))
proportion_women_survivors = number_women_survivors / number_women_onboard
proportion_men_survivors = number_men_survivors / number_men_onboard

print "Number of women survivors = %s" % number_women_survivors
print "Number of men suvivors = %s" % number_men_survivors
print "Survival rate of women = %s" % proportion_women_survivors
print "Survival rate of men = %s" % proportion_men_survivors
print 
