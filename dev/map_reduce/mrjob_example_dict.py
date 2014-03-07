import json
import time
start = time.time()
data = open('wc_input.txt').read()
data = data.split('\n')
data_out = {}
for d in data[:-1]:
    d = d.split(' ')
    for d2 in d:
        if d2 not in data_out:
            data_out[d2]=1
        else:
            data_out[d2]+=1

for d in data_out:
    print d, data_out[d]
end = time.time()
total = end-start
print 'Processed in %s seconds' % total
