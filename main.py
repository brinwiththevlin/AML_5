import math
import random
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


def main():
    random.seed(42)
    np.random.seed(42)
    # iris_data = load_iris()
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    A = breast_cancer_wisconsin_diagnostic.data.features.to_numpy()
    target = breast_cancer_wisconsin_diagnostic.data.targets.to_numpy()
    # A: np.ndarray = iris_data.data
    # target: np.ndarray = iris_data.target

    E = featureReduction(A)
    X_train, X_test, y_train, y_test = train_test_split(
        E, target, test_size=0.2, random_state=42)

    for k in ["linear", 2, 3, 4]:
        if k == "linear":
            clf = SVC(kernel="linear")
        else:
            clf = SVC(kernel="poly", degree=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        performanceEvaluation(y_test, y_pred)


def performanceEvaluation(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))

    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # class wise precision, recall and accuracy
    for i in range(conf_matrix.shape[0]):
        TP = conf_matrix[i][i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        TN = np.sum(conf_matrix) - TP - FP - FN

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)

        print(
            f"Class {i}: Precision = {precision}, Recall = {recall}, Accuracy = {accuracy}")


def featureReduction(A: np.ndarray) -> np.ndarray:
    # R = np.random.choice([1, 0, -1], size=(A.shape[1], 2), p=[1/6, 2/3, 1/6])
    R = np.random.choice([1, -1], size=(A.shape[1], 2))
    E = 1 / math.sqrt(2) * np.matmul(A, R)

    D1 = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            D1[i][j] = np.linalg.norm(A[i] - A[j])

    D2 = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            D2[i][j] = np.linalg.norm(E[i] - E[j])

    m = 1
    M = 1
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if D1[i][j] != 0 and D2[i][j] != 0:
                if D2[i][j] / D1[i][j] < m:
                    m = D2[i][j] / D1[i][j]
                if D2[i][j] / D1[i][j] > M:
                    M = D2[i][j] / D1[i][j]

    print(m, M)
    return E


if __name__ == "__main__":
    main()
