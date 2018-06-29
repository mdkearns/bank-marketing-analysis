# analysing bank marketing data using a Random Forest

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os

def main():
    # read data and extract X and y
    df = pd.read_csv('./bank-data/clean.csv', sep=';')
    values = df.values
    X = values[:, :-1]
    y = values[:, -1]

    sum = 0
    for val in y:
        if val == 0:
            sum += 1

    print('Naive Approach:', sum / len(y), '\t(Predict that no one subscribes)')

    # perform k-fold cross-validation
    kf = KFold(n_splits=5)

    # initialize a SVM classifier
    clf = LogisticRegression()

    ave = 0

    X, y = shuffle(X, y)

    for train_index, test_index in kf.split(X, y):
        # fit the SVM classifier
        clf.fit(X[train_index], y[train_index])
        ave += clf.score(X[test_index], y[test_index])

    ave /= 5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    clf.fit(X_train, y_train)
    print('Train: 85%; Test: 15%; Accuracy: ', clf.score(X_test, y_test))

    print('5-Fold Cross-Validation Accuracy:', ave)


if __name__ == '__main__':
    main()