#############################
#Example of api-based sentiment analysis using mashape
#Sentiment analysis of:
#http://www.cnn.com/2013/05/09/us/ohio-missing-women-found/
#http://www.happynews.com/

#Authentication:
#replace key with your api key

#Usage: 
#python mashape.py

#Documentation Examples:                                                                                                                        curl --include --request GET 'https://loudelement-free-natural-language-processing-service.p.mashape.com/nlp-text/?text=Friends%20and%20fellow%20sailors%20mourned%20double%20Olympic%20medalist%20Andrew%20%22Bart%22%20Simpson%20after%20the%20shocking%20news%20that%20he%20had%20died%20in%20San%20Francisco%20Bay%20while%20training%20for%20the%20America's%20Cup.' --header "X-Mashape-Authorization: iX7o6d4rTKAr4lcq1T209WJRp7aFQoJn"  

#curl --include --request GET 'https://loudelement-free-natural-language-processing-service.p.mashape.com/nlp-url/?url=http%3A%2F%2Fwww.cnn.com%2F2013%2F05%2F09%2Fus%2Fohio-missing-women-found%2F' --header "X-Mashape-Authorization: iX7o6d4rTKAr4lcq1T209WJRp7aFQoJn" 

#response = unirest.get("https://duckduckgo-duckduckgo-zero-click-info.p.mashape.com/?q=DuckDuckGo&callback=process_duckduckgo&no_html=1&no_redirect=1&skip_disambig=1&format=json",headers={"X-Mashape-Authorization": "iX7o6d4rTKAr4lcq1T209WJRp7aFQoJn"});
##############################

import urllib2
import json

KEY = open('key').read()

urls = ['http://www.cnn.com/2013/05/09/us/ohio-missing-women-found/','http://www.happynews.com/']

print 'running...\n'
for url in urls:
    url = url.replace('/','%2F').replace(':','%3A')
    url = 'https://loudelement-free-natural-language-processing-service.p.mashape.com/nlp-url/?url=%s' % url
    request = urllib2.Request(url)
    request.add_header("X-Mashape-Authorization", "Basic %s" % KEY)
    document  = []
    document = urllib2.urlopen(request).read()
    document = json.loads(document)
    print 'URL:',document['url-requested']
    print 'SENTIMENT:',document['sentiment-text']
    print 'SCORE:',document['sentiment-score']
    print 
