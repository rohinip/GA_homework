import json
import os

path = '/home/jacob/data/RecSys/MOVIEDATA/movie_dict2.json'

if not os.path.exists(path):
    out = {}
    data=open('/home/jacob/data/RecSys/MOVIEDATA/ml-1m/movies.dat').read()
    data = data.split('\n')
    for d in data:
        d = d.split('::')
        try:
            tmp = str(d[0])
            print d[0],d[1]
            if tmp not in out:
                out[tmp]=d[1].encode('utf8')
        except:pass
    #print out[str(1)]
    out = json.dumps(out)
    f = open(path,'w')
    f.write('%s' % str(out))
    f.close()
else:
    print 'file already exists'
