import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) 
data_train = pd.read_csv("asset/train.csv")
# print(data_train['engagement'][2])
for i in range(len(data_train['engagement'])) :
    if data_train["engagement"][i]==True :
        data_train['engagement'][i]=1
    else :
        data_train["engagement"][i]=0
        

X_data_train = data_train.iloc[:,1:-1]
y_data_train = data_train.iloc[:,-1]
y_data_train = y_data_train.astype('int')


def engagement_model():
    rec = None
    data_test = pd.read_csv("asset/test.csv")
    data_test_id = data_test.iloc[:,0]
    data_test = data_test.iloc[:,1:]
    
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score,roc_curve
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import GridSearchCV
    scaler = MinMaxScaler()
    X_train,X_test,y_train,y_test = train_test_split(X_data_train,y_data_train)
    # print(X_train.head())
    # print(y_train.head())
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
    data_test = scaler.transform(data_test)
    X_test_scale = scaler.transform(X_test)
    svm = SVC(probability=True,kernel='rbf',gamma=0.1,C=10)
    # param_choose = {'gamma':[0.01,0.1,1,10],'C' : [0.1,1,10]}
    # grid_acc=GridSearchCV(svm,param_grid=param_choose,scoring="roc_auc",cv=5)
    # grid_acc.fit(X_train_scale,y_train)
    # print(grid_acc.best_estimator_)
    # print(grid_acc.best_params_)
    # print(grid_acc.best_score_)
    rec = svm.predict_proba(data_test)
    rec=rec[:,1]
    rec = pd.Series(data=rec,index=data_test_id)
    return rec
    # raise NotImplementedError()
print(engagement_model())
