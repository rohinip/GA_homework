import os
import csv
import numpy as np

path = '/home/GA8/GA_homework/final/frenchopen.csv'
data = []

#Import our data
#The 'r' parameter is for read and the b parameter is for 'binary'
with open(path, 'rb') as csv_file_object:
  reader = csv.reader(csv_file_object)
  header = csv_file_object.next()
  for row in reader:
    data.append(row)

#http://www.tennis-data.co.uk/notes.txt
data = np.array(data)
tourney_number = data[0::,0]
location = data[0::,1]
tourney = data[0::,2]
date = data[0::,3]
court = data[0::,5] #indoor or outdoor
surface = data[0::,6] #clay, hard, carpet or grass
round = data[0::,7]
best_of = data[0::,8]
winner = data[0::,9]
loser = data[0::,10]
wrank = data[0::,11]
lrank = data[0::,12]
wpts = data[0::,13]
lpts = data[0::,14]
w1 = data[0::,15]
l1 = data[0::,16]
w2 = data[0::,17]
l2 = data[0::,18]
w3 = data[0::,19]
l3 = data[0::,20]
w4 = data[0::,21]
l4 = data[0::,22]
w5 = data[0::,23]
l5 = data[0::,24]
wsets = data[0::,25]
lsets = data[0::,26]
comment = data[0::,27]

#Number of upsets, based on player ranking
number_of_upsets = 0
i = 0
while i < wrank.size:
  if wrank[i] > lrank[i]:
    number_of_upsets += 1
  i += 1
print "i = %s" % i
print "number of upsets = %s" % number_of_upsets


#This works; shows the row# and comment for anything that wasn't completed
#index = 0
#for c in comment:
#  c = str(c)
#  index += 1
#  if c == "Completed":
#    continue
#  else: 
#    print "%s at row %s" % (c, index)
#    continue
