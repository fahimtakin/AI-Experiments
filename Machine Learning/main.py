from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

def irisLR():

    irisDataset = load_iris()


    print("Features")
    print(irisDataset.feature_names)


    X = irisDataset.data
    y = irisDataset.target
    print("Feature Matrix: ")
    print(X)
    print("Target names: ")
    print(list(irisDataset.target_names))
    print("Target: ")
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    lgObject = LogisticRegression(max_iter=3000, penalty='l2', solver='lbfgs', C=1.0)
    ModelParameters = lgObject.fit(X_train, y_train)

    trScore = ModelParameters.score(X_train, y_train)
    tsScore = ModelParameters.score(X_test, y_test)

    print("Accuracy on training (Logistic Regression): ", trScore)
    print("Accuracy on test (Logistic Regression): ", tsScore)

    print("Prediction on training set (Logistic Regression): ", ModelParameters.predict(X_train))

    # Ridge regularisation and Stochastic Gradient Descent algorithm as optimizer gives the highest accuracy on prediction with least overfitting and underfitting

def irisSGDC():

    irisDataset = load_iris()
    print("Features")
    print(irisDataset.feature_names)

    X = irisDataset.data
    y = irisDataset.target
    print("Feature Matrix: ")
    print(X.head())

    print("Target names: ")
    print(list(irisDataset.target_names))

    print("Target: ")
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    SGDCClassifier = SGDClassifier(loss='modified_huber', eta0=0.1, learning_rate='optimal', shuffle=False)

    SGD_model = SGDCClassifier.fit(X_train, y_train)

    SGD_trScore = SGD_model.score(X_train, y_train)
    SGD_tsScore = SGD_model.score(X_test, y_test)

    print("Accuracy on training (SGDCClassifier): ", SGD_trScore)
    print("Accuracy on test (SGDCClassifier): ", SGD_tsScore)

    print("SGDClassifier prediction (SGDCClassifier): ", SGD_model.predict(X_train))

    # ‘modified_huber’ a smooth loss function with optimal learning rate, and 0.1 initial learning rate eta is 0.1 and shuffle is False gives the highest accuracy for me


def breast_cancerLR():

    breastCancerDataset = load_breast_cancer()
    print("Features")
    print(breastCancerDataset.feature_names)

    X = breastCancerDataset.data
    y = breastCancerDataset.target



    print("Feature Matrix: ")
    print(X)

    print("Target names: ")
    print(list(breastCancerDataset.target_names))
    print("Target: ")
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    lgObject = LogisticRegression(max_iter=5000, penalty='l2', solver='newton-cg', C=1.0)
    ModelParameters = lgObject.fit(X_train, y_train)

    trScore = ModelParameters.score(X_train, y_train)
    tsScore = ModelParameters.score(X_test, y_test)

    print("Accuracy on training (Logistic Regression): ", trScore)
    print("Accuracy on test (Logistic Regression): ", tsScore)

    print("Prediction on training set (Logistic Regression): ", ModelParameters.predict(X_train))

    # Ridge regularisation and Newton-Conjugate Gradient algorithm as optimizer gives the highest accuracy on prediction with least overfitting and underfitting

def breast_cancerSGDC():

    breastCancerDataset = load_breast_cancer()
    print("Features")
    print(breastCancerDataset.feature_names)

    X = breastCancerDataset.data
    y = breastCancerDataset.target

    print("Feature Matrix: ")
    print(X)

    print("Target names: ")
    print(list(breastCancerDataset.target_names))
    print("Target: ")
    print(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    SGDCClassifier = SGDClassifier(loss='perceptron', eta0=0.1, learning_rate='optimal', shuffle=True, random_state=42)

    SGD_model = SGDCClassifier.fit(X_train, y_train)

    SGD_trScore = SGD_model.score(X_train, y_train)
    SGD_tsScore = SGD_model.score(X_test, y_test)

    print("Accuracy on training (SGDCClassifier): ", SGD_trScore)
    print("Accuracy on test (SGDCClassifier): ", SGD_tsScore)

    print("SGDClassifier prediction (SGDCClassifier): ", SGD_model.predict(X_train))

    # ‘perceptron’ loss function with optimal learning rate, and 0.1 initial learning rate eta is 0.1 and shuffle is true gives the highest accuracy for me



if __name__ == '__main__':

    #irisLR()
    #irisSGDC()
    breast_cancerLR()
    #breast_cancerSGDC()


