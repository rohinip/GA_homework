#################################
# Multithreading example script #
# General Assembly: DAT3        #
#################################

import threading
import time
import Queue

#iterations
number = 10

##duration
duration = .1

#single thread
print 'testing the single-threaded version'
start = time.time()
for a in range(0, number):
    print 'running cycle %s' % a
    time.sleep(duration)
end = time.time()
total = end-start
print 'finished single-threaded in %s seconds' % total

#function to be called in multi-threaded version
def run(a,q):
    string = 'running cycle %s' % a
    time.sleep(duration)
    q.put(string)

#multi-threaded
print '\ntesting the multi-threaded version'
#set up results queue
q = Queue.Queue()
start =time.time()

#create threads
threads=[]
for a in range(0,number):
    t = threading.Thread(target=run,args=(a,q,))
    threads.append(t)

#start threads
for t in threads:
    t.start()

#get results
result=[]
while True:
    if q.qsize()>0:
        string = q.get()
        result.append(string)
    else:
        pass
    if len(result)==number:
        break

#print results
for r in result:
    print r
end = time.time()
total =end-start
print 'finished multi-threaded in %s seconds' % total

