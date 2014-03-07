from roc import load_iris_data, cross_validate, knn, nb, lr, roc, svm

(XX,yy,y)=load_iris_data()

#classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb), ("Linear Regression",lr)]
classfiers_to_cv=[("Naive Bayes",nb)]
classfiers_to_cv=[('SVM',svm)]
ROC = True
for (c_label, classifer) in classfiers_to_cv :

    print
    print "---> Current classifier: %s <---" % c_label

    best_k=0
    best_cv_a=0
    for k_f in [2] :
        if ROC:
            outfile= 'roc.png'
            roc(XX, yy, classifer, k_f, outfile)
            print 'ROC plot file writen to %s' % outfile
            break
        if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

        print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)
    if not ROC:
        print "\n %s Highest Accuracy: fold <<%s>> :: <<%s>>\n" % (c_label, best_k, best_cv_a)
