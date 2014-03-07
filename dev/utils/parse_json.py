import json

data=[]

#Using "with" ensures that our file object closes correctly without
# us having to explicitly close it.

with open('/home/commons/yelp_valid_10biz.json', 'rb') as file_obj:
    data = json.loads(file_obj.read())

#The length of our list will tell use how many json objects (businesses) we loaded.
#If you are curious why we use len(data) instead of something like data.length, then read this:
# http://effbot.org/pyfaq/why-does-python-use-methods-for-some-functionality-e-g-list-index-but-functions-for-other-e-g-len-list.htm

length_of_list = len(data)

print
print "Loaded %s json objects" % length_of_list
print

#The following print statements illustrate how to get at the values we may want.
#Note that our "data" structure is a list of python dictionaries. So, data[x]
# gives us the dictionary (business, in this case) at index x.
# Within each dictionary, we can get at values by using the name of the key-value
# pair. Note that some keys have multiple values (e.g. categories).
#Spend some time with these examples to really understand how to get at the
# various elements of our data structure.

print "For item zero (first item) in the list of businesses, here are some key-value pairs:"
print "Yelp Business ID:", data[0]['business_id']
print "City:", data[0]['city']
print "State:", data[0]['state']
print "All Categories:", data[0]['categories']
print "Second Category:", data[0]['categories'][1]
print "Fourth Category:", data[0]['categories'][3]
print "Latitude:", data[0]['latitude']
print

print "For item five (sixth item) in the list of businesses, here are some key-value pairs:"
print "Yelp Business ID:", data[5]['business_id']
print "City:", data[5]['city']
print "State:", data[5]['state']
print "All Categories:", data[5]['categories']
print "First Category:", data[5]['categories'][0]
print "Third Catgory:", data[5]['categories'][2] #note this record does not have a fourth category, so index 3 willresult in an error
print "Latitude:", data[5]['latitude']
print
