import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
cancer = load_breast_cancer()

def answer_zero():
    X=cancer.data
    Y=cancer.target
    return len(X[0])
def answer_one():
    Z_name = cancer.feature_names
    Z_name = list(Z_name)
    Z_name.append("target")
    cancer_data = cancer.data
    cancer_target = cancer.target 
    cancert =  cancer_target.reshape(569,1)
    # print(cancer_data)
    # print(cancert)
    
    C=np.hstack((cancer_data,cancert))

    Data_frame = pd.DataFrame(data = C , columns=Z_name)
   
    print(Data_frame)
def answer_two() :
    cancer_target = cancer.target
    mal_tar=0
    beg_tar=0
    for i in cancer_target : 
        if i ==0: 
            beg_tar+=1
        else :
            mal_tar+=1
    tar_class = pd.Series([beg_tar,mal_tar],index=['malignant', 'benign'])
    print(tar_class)

def answer_three():
    cancer_data=cancer.data
    cancer_target=cancer.target
    X=pd.DataFrame(data=cancer_data,columns=cancer.feature_names)
    y=pd.Series(data=cancer_target)
   
    return (X,y)
def answer_four():
    cancer_data=cancer.data
    cancer_target = cancer.target
    X_train,X_test,Y_train,Y_test = train_test_split(cancer_data,cancer_target,train_size=426,random_state=0)
    return (X_train,X_test,Y_train,Y_test)
def answer_five():
    cancer_data=cancer.data
    cancer_target = cancer.target
    X_train,X_test,Y_train,Y_test = train_test_split(cancer_data,cancer_target,train_size=426,random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,Y_train)
    return knn
def answer_six():
    # YOUR CODE HERE
    cancer_data=cancer.data
    cancer_target = cancer.target
    X_train,X_test,Y_train,Y_test = train_test_split(cancer_data,cancer_target,train_size=426,random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,Y_train)
    cancer_data_ = pd.DataFrame(data=cancer_data,columns=cancer.feature_names)
    print(cancer_data_)
    print(cancer_data_.mean().values.reshape(1, -1))
def answer_seven():
        cancer_data=cancer.data
        cancer_target = cancer.target
        X_train,X_test,Y_train,Y_test = train_test_split(cancer_data,cancer_target,train_size=426,random_state=0)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train,Y_train)
        print(knn.predict(X_test))
def answer_eight():
    cancer_data=cancer.data
    cancer_target = cancer.target
    X_train,X_test,Y_train,Y_test = train_test_split(cancer_data,cancer_target,train_size=426,random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,Y_train)
    knn_score=knn.score(X_test,Y_test)
    return knn_score
    # raise NotImplementedError()
def accuracy_plot():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Using k=5 for the example
    knn_classifier.fit(X_train, y_train)

    # Make predictions on train and test sets
    train_predictions = knn_classifier.predict(X_train)
    test_predictions = knn_classifier.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Prepare data for the plot
    data_points = ['Train Set', 'Test Set']
    prediction_scores = [train_accuracy, test_accuracy]

    # Create the plot
    plt.bar(data_points, prediction_scores)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Test Prediction Scores (k-Nearest Neighbors)')
    plt.ylim(0.9, 1.0)  # Set the y-axis limits for better visualization
    plt.grid(True)

    plt.show()



