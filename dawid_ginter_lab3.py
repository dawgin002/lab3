import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Zadanie 1 Ładowanie danych treningowych i testotwych
X_test = np.loadtxt("X_test.txt", delimiter=" ")
X_train = np.loadtxt("X_train.txt", delimiter=" ")
y_test = np.loadtxt("y_test.txt", delimiter=" ")
y_train = np.loadtxt("y_train.txt", delimiter=" ")

# Zadanie 2

def svmClassifier():
    clf = svm.SVC(kernel='linear')
    clf = clf.fit(X_train, y_train)
    clf.predict(X_test)


def knnClassifier(n_neighbors: int):
    scores_list = []
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    

def decisionTreeClsf():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

def randomForestClsf(max_depth, random_state):
    clf = RandomForestClassifier(max_depth, random_state)
    clf.fit(X_train, y_train)

# Zadanie 3

def confMatrix(clf):
    metrics.plot_confusion_matrix(clf, X_test, y_test,
                            cmap=plt.cm.Blues,
                            normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.show()

def accuracyScore(y_pred):
    print(metrics.accuracy_score(y_test, y_pred))

def recall(y_pred):
    print(metrics.precision_recall_fscore_support(y_test, y_pred, average='macro'))

def f_one(y_pred):
    print(metrics.f1_score(y_test, y_pred, average='macro'))

def auc(y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
    print(metrics.auc(fpr, tpr))

def main():
    svmClassifier()

if __name__ == "__main__":
    main()

