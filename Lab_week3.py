import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
dataset = load_digits()

X,y = dataset.data,dataset.target


y_binary_imbalance = y.copy()
y_binary_imbalance[y_binary_imbalance!=1] = 0

X_train,X_test,y_train,y_test = train_test_split(X,y_binary_imbalance,random_state=0)
#1 confusion_matrix
def func1() :
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix

    svm = SVC(kernel='rbf',C=1).fit(X_train,y_train)
    svm_predicted = svm.predict(X_test)
    confusion = confusion_matrix(y_test,svm_predicted)
    print("SVC with rbs kernel")
    print(confusion)
    print(svm.score(X_test,y_test))

def func2() :
    from sklearn.dummy import DummyClassifier
    dummy_majoity = DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
    print(dummy_majoity.predict(X_test))
    print(dummy_majoity.score(X_test,y_test))
def func3():
    from sklearn.metrics import confusion_matrix
    from sklearn.dummy import DummyClassifier
    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
    y_imbalance_predict = dummy_majority.predict(X_test)
    confusion  = confusion_matrix(y_test,y_imbalance_predict)
    print(confusion)
def func4() :
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix

    svm = SVC(kernel='linear',C=1).fit(X_train,y_train)
    svm_predicted = svm.predict(X_test)
    confusion = confusion_matrix(y_test,svm_predicted)
    print("SVC with linear kernel")
    print(confusion)
    print(svm.score(X_test,y_test))
    return svm_predicted
#2 accuracy, percision,recall f1 score
def func5():
    svm_precict = func4()
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score,precision_score,accuracy_score
    print('Recall score {:.2f}'.format(recall_score(y_test,svm_precict)))
    print('Percision score{:.2f}'.format(precision_score(y_test,svm_precict)))
    print("Accuracy score {:.2f}".format(accuracy_score(y_test,svm_precict)))
    print(classification_report(y_test,svm_precict,target_names=['1','not 1']))

def func6() :
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(X_train,y_train)
    y_score_lr = lr.decision_function(X_test)
    # this function caculate the distance from actual data to predict data, and 
    # when we estimate the threshold, we can base on this 
    y_score_list = list(zip(y_test[0:20],y_score_lr[0:20]))
    
    return y_score_lr
def func7():
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression().fit(X_train,y_train)
    # https://chat.openai.com/share/9e93b59d-3def-4033-b3ab-5e9667879ad2?fbclid=IwAR0BKpAqrTttka7NkadEgzXlrc7G10PPDxugMeaR-7ANiPifodtwr3v67o0
    y_proba_lr = lr.predict_proba(X_test)
    y_proba_list = list(zip(y_test[0:20],y_proba_lr[0:20]))
    y_data_frame = pd.DataFrame(data=y_proba_list)
    print(y_data_frame)
def func8() :
    from sklearn.metrics import precision_recall_curve
    y_score_lr = func6()
    precision, recall, threshold = precision_recall_curve(y_test,y_score_lr)

    closest_zero = np.argmin(np.abs(threshold))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    # plt.axes().set_aspect('equal')
    plt.show()
#3 ROC curves, AUC 
def func9() :
    from sklearn.metrics import roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalance, random_state=0)
    lr = LogisticRegression()
    y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    # plt.axes().set_aspect('equal')
    plt.show()
def func10() :
    from matplotlib import cm
    from sklearn.svm import SVC
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalance, random_state=0)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    for g in [0.01, 0.1, 0.20, 1]:
        svm = SVC(gamma=g).fit(X_train, y_train)
        y_score_svm = svm.decision_function(X_test)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        accuracy_svm = svm.score(X_test, y_test)
        print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, 
                                                                        roc_auc_svm))
        plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
                label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
    plt.legend(loc="lower right", fontsize=11)
    plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
    # plt.axes().set_aspect('equal')

    plt.show()
#4 Grid search and cross_validation
# grid search is function to find the best parameter for the model
# cross_val_score ís function that return the score on each fold of data
def func11() :

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    dataset = load_digits()
    X, y = dataset.data, dataset.target == 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = SVC(kernel='rbf')
    grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

    # default metric to optimize over grid parameters: accuracy
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
    grid_clf_acc.fit(X_train, y_train)
    y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

    print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
    print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

    # alternative metric to optimize over grid parameters: AUC
    grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 
    
    print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)
def func12():
    from sklearn.metrics import precision_recall_curve
    y_score_lr = func6()
    precision, recall, threshold = precision_recall_curve(y_test,y_score_lr)
    for i in range (len(recall)) :
        if recall[i] == 0.8:
            print(precision[i])
func9()