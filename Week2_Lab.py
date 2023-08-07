import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

def intro():
    

    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    a=[1,3,6,9]
    ans = np.zeros((4,100))
    
    for i in range(len(a)):
        poly = PolynomialFeatures(degree=a[i])
        X_train_=X_train.reshape(-1,1)
        Y_train=y_train.reshape(-1,1)
        X_poly=poly.fit_transform(X_train_)
        lr=LinearRegression().fit(X_poly,Y_train)
        x=np.linspace(0,10,100).reshape(-1,1)
        x_=poly.transform(x)
        x_predict=lr.predict(x_)
        x_predict=x_predict.reshape(1,-1)
        ans[i]=x_predict
    print(ans)
    return ans
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()
def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score

    r2_train = np.array([])
    r2_test = np.array([])
    X_test_=X_test.reshape(-1,1)
    Y_test=y_test.reshape(-1,1)
    X_train_=X_train.reshape(-1,1)
    Y_train=y_train.reshape(-1,1)
    for i in range(0,10) :
        poly= PolynomialFeatures(degree=i)
        X_poly=poly.fit_transform(X_train_)
        lr = LinearRegression().fit(X_poly,Y_train)
        X_tet_=poly.transform(X_test_)
        r_score_train = lr.score(X_poly,Y_train)
        r_score_test = lr.score(X_tet_,Y_test)
        r2_train = np.append(r2_train,r_score_train)
        r2_test=np.append(r2_test,r_score_test)
    return (tuple((r2_train,r2_test)))
    # raise NotImplementedError()

def answer_three() :
    a = answer_two()
    r_train = a[0]
    r_test=a[1]
    print(r_train)
    print(r_test)
    x_ = [1,2,3,4,5,6,7,8,9,10]
    plt.figure()
    plt.plot(x_,r_train,'o',label='R^2 score training',markersize=10)
    plt.plot(x_,r_test,'o',label='R^2 score testing',markersize=10)
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()
    return(0,9,6)
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression,Lasso
    X_train_=X_train.reshape(-1,1)
    Y_train=y_train.reshape(-1,1)
    X_test_=X_test.reshape(-1,1)
    Y_test=y_test.reshape(-1,1)
    ls = Lasso(alpha=0.01 , max_iter=10000,tol=0.1).fit(X_train_,Y_train)
    Lasso_r2_test_score = ls.score(X_test_,Y_test)
    poly = PolynomialFeatures(degree=12)
    X_poly= poly.fit_transform(X_train_)
    lr = LinearRegression().fit(X_poly,Y_train)
    X_tet = poly.transform(X_test_)
    Linear_r2_test_score = lr.score(X_tet,Y_test)
    return (Linear_r2_test_score,Lasso_r2_test_score)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('asset/mushroom/agaricus-lepiota.data')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)
def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2,y_train2)
    

    # Get the feature importances
    feature_importances = clf.feature_importances_
    feature_names = clf.feature_names_in_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    sort_df = feature_importance_df.sort_values(by="Importance",ascending=False)
    top_5 = sort_df['Feature'].iloc[:5]
    # Get the 5 most important features
    top_5=list(top_5)
    return top_5
    # Print or use the names of the top 5 important features
    
def Plot_for_answer_5():
    from sklearn.tree import DecisionTreeClassifier
    from adspy_shared_utilities import plot_decision_tree
    from adspy_shared_utilities import plot_feature_importances
    clf = DecisionTreeClassifier().fit(X_train2,y_train2)
    y_value = y_mush.values
    y_values = []
    for i in y_value:
        if i == True :
            y_values.append("True")
        else :
            y_values.append("False")
    
    plot_decision_tree(clf,X_mush.columns,y_values)
    plt.figure(figsize=(10,6),dpi=80)
    plot_feature_importances(clf, X_mush.columns)
    plt.tight_layout()
    plt.show()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    range_gamma = np.logspace(-4,1,6)
    train_score,valid_score = validation_curve(SVC(),X_mush,y_mush,param_name="gamma",param_range=range_gamma,cv=3,n_jobs=2)
    train_score_mean_ave = np.array([])
    test_score_mean_ave = np.array([])
    for i in range (0,6) :
        tmp_train_score = np.mean(train_score[i])
        tmp_test_score =np.mean(valid_score[i])
        train_score_mean_ave = np.append(train_score_mean_ave,tmp_train_score)
        test_score_mean_ave = np.append(test_score_mean_ave,tmp_test_score)
    train_score_mean_ave = train_score_mean_ave.reshape(6,)
    test_score_mean_ave = test_score_mean_ave.reshape(6,)
    return (train_score_mean_ave,test_score_mean_ave)
def answer_seven(): 
    range_gamma = np.logspace(-4,1,6)
    plt.figure()

    train_score =np.array([0.89838749, 0.98104382, 0.99895372, 1.        , 1.        ,1.        ])
    test_score =np.array([0.88749385, 0.82951748, 0.84170359, 0.86582964, 0.83616445, 0.51797144])
    train_score = train_score.reshape(-1,1)
    test_score = test_score.reshape(-1,1)
    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(train_score,marker= 'o',color= "darkorange")
    plt.plot(test_score,marker="o",color="navy")
    plt.semilogx(range_gamma,train_score,label = "Training Score",color = "darkorange",lw=lw)
    plt.semilogx(range_gamma,test_score,label="Valid Score",color="navy",lw=lw)
    plt.legend(loc="best")
    plt.show()
    return(0.0001,10,0.1)


    
