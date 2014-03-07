#!/usr/local/bin/python
print 'importing modules'
import pdb
import json
import recsys.algorithm
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE
from recsys.utils.svdlibc import SVDLIBC

"""
##install python-recsys
git clone https://github.com/python-recsys/python-recsys.git
cd python-recsys/
python setup.py build
python setup.py install

##install divisi2
sudo pip install divisi2

#get data
###go to your working directory
mkdir MOVIEDATA
#download http://www.grouplens.org/system/files/ml-1m.zip
#put the resulting directory in MOVIEDATA
"""

names = json.loads(open('/home/commons/RecSys/MOVIEDATA/movie_dict2.json').read())

def SVDloadData():
    svd = SVD()
    recsys.algorithm.VERBOSE = True
    dat_file = '/home/commons/RecSys/MOVIEDATA/MOVIEDATA/ml-1m/ratings.dat'
    svd.load_data(filename=dat_file, sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})
    print svd.get_matrix()
    return svd

def SVDcompute(svd):
    k=100
    svd.compute(k=k, min_values=10, pre_normalize=None, mean_center=True,post_normalize=True)
    return svd

def SVDgetSimilar(svd, ITEMID1):
    simMovie = svd.similar(ITEMID1)
    for ind,score in simMovie:
        ind = str(ind)
        print 'similar movie: %s' % names[ind]

def SVDpredict(ITEMID, USERID, MIN_RATING, MAX_RATING):
    pred = svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
    actual = svd.get_matrix().value(ITEMID, USERID)
    print 'predicted rating = {0}'.format(pred)
    print 'actual rating = {0}'.format(actual)

def SVDrecommend(ITEMID):
    recMovie = svd.recommend(ITEMID)
    for ind,score in recMovie:
        print 'user %s' % ind

def SVDloadData2():
    dat_file='/home/commons/RecSys/MOVIEDATA/ml-1m/ratings.dat'
    pct_train=0.5
    data = Data()
    data.load(dat_file, sep='::', format={'col':0, 'row':1, 'value':2,'ids':int})
    return data

def SVDtrain2(data,pct_train):
    train, test = data.split_train_test(percent=pct_train)               
    K=100
    svd = SVD()
    svd.set_data(train)
    svd.compute(k=K, min_values=5, pre_normalize=None, mean_center=True,
    post_normalize=True)
    return svd,train,test

def SVDtest2(data,train,test,pct_train):

    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():
        try:
            pred_rating = svd.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s\n' % mae.compute()

ITEMID = 1      # toy story                                                 
#ITEMID = 1221   # godfather II 

MIN_RATING = 0.0
MAX_RATING = 5.0
USERID = 1

pct_train=0.5

print 'loading data'
svd = SVDloadData()
print dog
print 'computing svd'
svd = SVDcompute(svd)   
print '\ngetting similar titles %s' % names[str(ITEMID)]  
SVDgetSimilar(svd, ITEMID)
print '\npredicting rating'                             
SVDpredict(ITEMID, USERID, MIN_RATING, MAX_RATING)
print '\nfinding users to recommend %s' % names[str(ITEMID)]                                                                                        
SVDrecommend(ITEMID)
"""
print '\nloading data set 2'
data = SVDloadData2()
print 'training'
svd,train,test = SVDtrain2(data,pct_train)
print 'evaluating performance'
SVDtest2(svd,train,test,pct_train)
print 'done!'
"""
