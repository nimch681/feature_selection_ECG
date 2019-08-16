from sklearn import svm
from codes.python import metric

## small C from 1 to 3 seems to be good for f class recalled
## degree 3 is better all around
## Tol at 0.01 seems best

def svm_model_poly(X_train, y_train, X_test, y_test, C=10, kernel='poly', degree=3, gamma='auto', 
                        coef0=0.001, shrinking=True, probability=True, tol=0.01, 
                        cache_size=200, verbose=False, 
                        max_iter=-1, decision_function_shape='ovo', random_state=None, jk = False,labels = [0,1,2,3]):
    
    svm_model_linear = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                        coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, 
                        cache_size=cache_size, verbose=verbose, 
                        max_iter=max_iter, decision_function_shape='ovo', random_state=random_state) 
    
    svm_model_linear.fit(X_train, y_train)
    y_pred = svm_model_linear.predict(X_test)    
    print(confusion_matrix(y_test,y_pred, labels=labels))  
    print(classification_report(y_test,y_pred))
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
        
    
    return svm_model_linear,y_pred, met


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def Linear_D(X_train, y_train, X_test, y_test,jk = False, labels = [0,1,2,3]):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)  
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred, labels=labels))  
    print(classification_report(y_test,y_pred, labels=labels))
    
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    return clf,y_pred, met

from sklearn.ensemble import RandomForestClassifier

## Number of feature at 50 or 100 is good too
## numner of estimators leaves at 1000 

def randomForest(X_train, y_train, X_test, y_test,jk = False,n_estimators=1000, criterion='gini', max_depth=16, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0001,min_impurity_decrease=0.0, max_features=50, max_leaf_nodes=None, class_weight='balanced', labels = [0,1,2,3]):
    
    rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf, class_weight=class_weight, min_impurity_decrease=min_impurity_decrease,max_features=max_features)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)    
    print(confusion_matrix(y_test,y_pred, labels=labels))  
    print(classification_report(y_test,y_pred, labels=labels))
    
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    return rf, y_pred, met


### xgbo is good for normal classes too
## do this one next 
## XGB is very goos for V class
## XGboost tree classifcation is good with 5 or more tree depth
from xgboost import XGBClassifier


def xgboost(X_train,y_train, X_test, y_test, jk = False,max_depth = 8, eta = 1, gamma = 0.001, min_child_weight=0.1,max_delta_step=10, subsample=0.6,colsample_bytree = 0.7, colsample_bylevel= 1, colsample_bynode=1, alpha=0, reg_lambda= 1, tree_method='exact', grow_policy='depthwise', refresh_leaf=True, process_type='default', predictor= 'cpu_predictor', labels = [0,1,2,3]): 
    model = XGBClassifier(max_depth = max_depth, eta = eta, gamma = gamma,min_child_weight=min_child_weight,subsample=subsample, max_delta_step= max_delta_step,colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,colsample_bynode=colsample_bynode,alpha=alpha,reg_lambda=reg_lambda,grow_policy=grow_policy, tree_method=tree_method,refresh_leaf=refresh_leaf,process_type=process_type, predictor= predictor,objective = 'reg:logistic')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    print(confusion_matrix(y_test.ravel(),y_pred.ravel(), labels=labels)) 
    print(classification_report(y_test,y_pred, labels=labels))

    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    return model, y_pred,met



from sklearn.linear_model import LogisticRegression
## L1 one is good , l2 would be used
## very good at f class recall
## solver = ['newton-cg', 'lbfgs', 'liblinear'are all good, ensemble with all would be good

## false dual is better
def logisticRegress(X_train, y_train, X_test, y_test,jk = False, penalty='l2', dual=False, tol=0.0001, C=100, fit_intercept=True, intercept_scaling=10, class_weight='balanced', random_state=None, solver='liblinear', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, labels = [0,1,2,3]):
    lr = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    lr.fit(X_train, y_train.ravel())  
    y_pred = lr.predict(X_test)
    print(confusion_matrix(y_test.ravel(),y_pred.ravel(), labels=labels))  
    print(classification_report(y_test.ravel(),y_pred.ravel(), labels=labels))
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    
    return lr,y_pred, met



from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

def ada(X_train, y_train, X_test, y_test,jk = False,labels = [0,1,2,3]):

    svm_model_poly = svm.SVC(C=10,  kernel='linear', degree=3, gamma='auto', 
                        coef0=0.0, shrinking=True, probability=True, tol=0.1, 
                        cache_size=200, verbose=False, 
                        max_iter=-1, decision_function_shape='ovo', random_state=None)

    ada = AdaBoostClassifier(base_estimator=svm_model_poly, n_estimators=50 )
    ada.fit(X_train, y_train)  
    y_pred = ada.predict(X_test)
    print(confusion_matrix(y_test.ravel(),y_pred.ravel(), labels=labels))  
    print(classification_report(y_test.ravel(),y_pred.ravel(), labels=labels))
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    
    return ada, y_pred, met


from sklearn import svm

## Number of feature at 50 or 100 is good too


def svm_model_linear(X_train, y_train, X_test, y_test,jk = False, C=10, kernel='linear', degree=3, gamma='auto', 
                        coef0=0.0, shrinking=True, probability=True, tol=0.1, 
                        cache_size=200, verbose=False, 
                        max_iter=-1, decision_function_shape='ovo', random_state=None, labels = [0,1,2,3] ):
    
    svm_model_linear = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                        coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, 
                        cache_size=cache_size, verbose=verbose, 
                        max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state) 
    
    svm_model_linear.fit(X_train, y_train)
    y_pred = svm_model_linear.predict(X_test)    
    print(confusion_matrix(y_test.ravel(),y_pred.ravel(), labels=labels))  
    print(classification_report(y_test.ravel(),y_pred.ravel(), labels=labels))
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    return svm_model_linear, y_pred, met




from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier
import time
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm



def voting_ensemble(X_train, y_train, X_test,y_test, jk= False,labels=[0,1,2,3]):
    svm_model_linear = svm.SVC(C=10, kernel='linear', degree=3, gamma='auto', 
                            coef0=0.0, shrinking=True, probability=True, tol=0.1, 
                            cache_size=200, verbose=False, 
                            max_iter=-1, decision_function_shape='ovo', random_state=None)

    svm_model_poly = svm.SVC(C=10, kernel='poly', degree=3, gamma='auto', 
                            coef0=0.001, shrinking=True, probability=True, tol=0.01, 
                            cache_size=200, verbose=False, 
                            max_iter=-1, decision_function_shape='ovo', random_state=None)


    rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=16, min_samples_split=2, 
                                min_samples_leaf=3, min_weight_fraction_leaf=0.0001,min_impurity_decrease=0.0, 
                                max_features=50, max_leaf_nodes=None, class_weight='balanced')


    lr = LogisticRegression( penalty='l2', dual=False, tol=0.0001, C=100, fit_intercept=True, intercept_scaling=10, 
                            class_weight='balanced', random_state=None, solver='warn', max_iter=100, multi_class='warn',
                            verbose=0, warm_start=False, n_jobs=None)


    xgb = XGBClassifier(max_depth = 8, eta = 1, gamma = 0.001, min_child_weight=0.1,max_delta_step=10, 
                        subsample=0.6,colsample_bytree = 0.7, colsample_bylevel= 1, colsample_bynode=1, alpha=0, 
                        reg_lambda= 1, tree_method='exact', grow_policy='depthwise', refresh_leaf=True, 
                        process_type='default', predictor= 'cpu_predictor',objective = 'reg:logistic')




    eclf = VotingClassifier(estimators=[ ('xgb',xgb), ('lr',lr), ('svm_ln',svm_model_linear)], voting='hard')#, weights=[2,2,2,2])



    #clf1 = clf1.fit(X, y)
    #clf2 = clf2.fit(X, y)
    #clf3 = clf3.fit(X, y)
    eclf = eclf.fit(X_train, y_train)

    predict = eclf.predict(X_test)
    #scores = cross_val_score(X_train, y_train.data, iris.target, cv=10)
    #scores.mean()                             

    print(confusion_matrix(y_test,predict, labels=labels))  
    print(classification_report(y_test,predict, labels=labels))
    met = None
    if jk == True:
        met = metric.get_metrics(y_pred,y_test,lb=labels)
        print(met)
    
    return eclf, predict, met