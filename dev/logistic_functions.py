from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#KNN=True 
#NB=False

def load_iris_data() :

    # load the iris dataset from the sklearn module
    iris = datasets.load_iris()

    # extract the elements of the data that are used in this exercise
    return (iris.data, iris.target, iris.target_names)

def logitr(X_train, y_train, reg_param):
    # function returns a Logistic Regression (logitr) object

    clf = LogisticRegression(C=reg_param)
    clf.fit(X_train, y_train)

    return clf

def lr(X_train, y_train):
    # funtion returns a Linear Regression (lr) object
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
def cross_validate(XX, yy, classifier, k_fold) :  #I think you need to add C as a param here

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
