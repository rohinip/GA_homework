import urllib
import os
import json
dir_out = '/home/jacob/yahoo_data_20130110'

if not os.path.exists(dir_out):
    print 'Creating directory %s' % dir_out
    os.mkdir(dir_out)

#symbols = ['FB','GOOG','AAPL']
#symbols = ['FB']
symbols  = json.loads(open("symbols.json").read())
for symbol in symbols:
    data_out = {}
    if 'more than 25' not in data_out:
        data_out['more than 25']=0
    if 'less than 25' not in data_out:
            data_out['less than 25']=0
    try:
        url = 'http://ichart.finance.yahoo.com/table.csv?s=%s&d=6&e=23&f=2013&g=d&a=8&b=7&c=1984&ignore=.csv' % symbol
        data = urllib.urlopen(url).read()
        file = '%s/%s_data.csv' % (dir_out,symbol)
        f = open(file,'w')
        f.write('%s' % str(data))
        f.close()
        print 'wrote file %s' % file
        data = data.split('\n')
        for d in data:
            d = d.split(',')
            try:
                close = d[6]
                if float(close) >= float(25):
                    data_out['more than 25']+=1
                elif float(close) < float(25):
                    data_out['less than 25']+=1
            except:pass
        print symbol,':', json.dumps(data_out)
    except:
        print 'failed for %s' % symbol
