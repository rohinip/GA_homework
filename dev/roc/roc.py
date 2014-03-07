from sklearn import datasets
from sklearn  import svm as SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
random_state = np.random.RandomState(0)

KNN=True 
NB=False
def load_iris_data() :

    # load the iris dataset from the sklearn module
    iris = datasets.load_iris()

    # extract the elements of the data that are used in this exercise
    return (iris.data, iris.target, iris.target_names)

def svm(X_train,y_train):
    clf = SVM.SVC(kernel='linear', probability=True, random_state=0)
    clf.fit(X_train, y_train)
    return clf

def lr(X_train, y_train):
    # funtion returns an LR object
    #  useful methods of this object for this exercise:                                                                                                                    
    #   fit(X_train, y_train) --> fit the model using a training set                                                                                                       
    #   predict(X_classify) --> to predict a result using the trained model                                                                                                
    #   score(X_test, y_test) --> to score the model using a test set
    
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    return clf

def knn(X_train, y_train, k_neighbors = 3 ) :
    # function returns a kNN object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf


def nb(X_train, y_train) :
    # this function returns a Naive Bayes object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

# generic cross validation function
def cross_validate(XX, yy, classifier, k_fold) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :

        model = classifier(XX[[ train_slice  ]],
                         yy[[ train_slice  ]])

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold

def roc(XX, yy, classifier, k, outfile):

    XX, yy = XX[yy != 2], yy[yy != 2] # make binary
    n_samples, n_features = XX.shape
    XX = np.c_[XX, random_state.randn(n_samples, 150 * n_features)]###add some noise to make the problem harder
    XX, yy = shuffle(XX, yy, random_state=random_state)

    k_fold_indices = KFold(len(XX), n_folds=k, indices=True, shuffle=True, random_state=0)
    
    thresholds = range(1,100)
    fpr_result=[]
    tpr_result=[]
    for t in thresholds:
        scores_test = []
        scores_train = []
        for train_slice, test_slice in k_fold_indices:
            
            model = classifier(XX[[ train_slice  ]], yy[[ train_slice  ]])
            
            predictions = model.predict_proba(XX[[ test_slice ]])
            scores = yy[[ test_slice ]]
            
            for p in predictions:
                scores_test.append(p[1]*float(100))
            for s in scores:
                scores_train.append(s)
            fp=0
            tp=0
            fn=0
            tn=0
            for a in range(len(scores_test)):
                if scores_test[a]>=float(t) and scores_train[a]==1:
                    tp+=1
                if scores_test[a]>=float(t) and scores_train[a]==0:
                    fp+=1
                if scores_test[a]<float(t) and scores_train[a]==1:
                    fn+=1
                if scores_test[a]<float(t) and scores_train[a]==0:
                    tn+=1

        try:tpr_result.append(float(tp)/(float(tp)+float(fn)))
        except:tpr_result.append(0)
        try:fpr_result.append(float(fp)/(float(fp)+float(tn)))
        except:fpr_result.append(0)
    fig, ax = plt.subplots(1)
    ax.plot(fpr_result,tpr_result)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.savefig(outfile, bbox_inches='tight')
