import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('asset/fraud_data.csv')
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv('asset/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
def answer_one():
    data_class = data.iloc[:,29]
    data_class=np.array(data_class)
    print(type(data_class))
    k=0
    sum1=0
    for i in data_class :
        sum1+=1
        if i==1 : 
            k+=1

    print(k)
    print(sum1)
    print(data.iloc[16269,:])
def answer_two():
    # Use X_train, X_test, y_train, y_test for all of the following questions
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    dummy = DummyClassifier().fit(X_train,y_train)
    y_predict = dummy.predict(X_test)
    recall_sco = recall_score(y_test,y_predict)
    print(dummy.score(X_test,y_test))
    print(recall_sco)    
def answer_three() :
    from sklearn.svm import SVC
    from sklearn.metrics import recall_score,precision_score
    svm = SVC().fit(X_train,y_train)
    y_predict = svm.predict(X_test)
    recall_sco = recall_score(y_test,y_predict)
    precision_sco = precision_score(y_test,y_predict)
    accuracy_sco = svm.score(X_test,y_test)
    return (accuracy_sco,recall_sco,precision_sco)
def answer_four() :
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    threshold = -220
    svm = SVC(C=1e9,gamma=1e-07).fit(X_train,y_train)

    check  = svm.decision_function(X_test)
    predic_with_thres = []
    for i  in check :
        if i >=threshold :
            predic_with_thres.append(1)
        else :
            predic_with_thres.append(0)
    predic_with_thres = np.array(predic_with_thres)
    con = confusion_matrix(y_test,predic_with_thres)

    print(con)
def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve,roc_curve
    from sklearn.metrics import roc_auc_score
    lr = LogisticRegression().fit(X_train,y_train)
    y_decision = lr.decision_function(X_test)

    precision,recall,threshold = precision_recall_curve(y_test,y_decision)
    FPR,TPR,threshold_score = roc_curve(y_test,y_decision)
    roc = roc_auc_score(y_test,y_decision)

    fig,ax = plt.subplots()
    
    plt.figure()
    plt.xlim([0.0,1.01])
    plt.ylim([0.0,1.01])
    plt.plot(precision,recall,label="Precision Recall",color="navy")
    plt.axvline(x=0.75)
    plt.xlabel("Precision")
    plt.ylabel("recall")
    plt.show()

    threshold_index = abs(threshold - 0.75).argmin()
    intersection_precision = precision[threshold_index]
    intersection_recall = recall[threshold_index]
    print(intersection_precision)
    print(intersection_recall)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(FPR, TPR, lw=3)
    plt.axvline(x=0.16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.show()
    
    return (0.825,0.934)
    
def answer_six():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    from sklearn.model_selection import cross_val_score
    param_choose = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10]}
    lr = LogisticRegression(solver='liblinear')
    grid_lr_acc = GridSearchCV(lr,param_grid=param_choose,scoring="recall",cv=3)
    grid_lr_acc.fit(X_train,y_train)
    tmp = grid_lr_acc.cv_results_
    data_ans = pd.DataFrame(data=tmp)
    mean_test_score = tmp['mean_test_score']
    result = np.array(mean_test_score).reshape(4,2)
    print(result)
answer_five()

