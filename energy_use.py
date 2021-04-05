# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:47:51 2021

@author: emesh
"""


import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("C:\EnergyOptimization\MunicipalEnergyUse.csv")
    #X = df.drop("Total", axis=1).values
    X = df.iloc[:,0:len(df.columns)-1].values
    #Y = df.Total.values
    Y = df.iloc[:,-1].values
    rf = ensemble.RandomForestClassifier(n_jobs=-1)
 
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300, 400], 
        "max_depth": [1, 3, 5, 7],
        "criterion": ["gini", "entropy"],
    }
    
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=5,
        n_jobs=1
    )
    model.fit(X, Y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())

#data = pd.read_csv('MunicipalEnergyUse.csv', sep=',')
#x_var=data.drop(['Class'], axis=1)
#y_var=data['Class']
#xTrain, xValid, yTrain, yValid=train_test_split(x_var, y_var, test_size=0.3, random_state=4)
    
#Randomized Search
    
if __name__ == "__main__":
    df = pd.read_csv("C:\EnergyOptimization\MunicipalEnergyUse.csv")
    X = df.drop("Total").values
    Y = df.Total.values
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20, 1),
        "criterion": ["gini", "entropy"],
    }
    
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1
        cv=5,
    )
    model.fix(X, Y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())


if __name__ == "__main__":
    df = pd.read_csv("C:\EnergyOptimization\MunicipalEnergyUse.csv")
    X = df.drop("Total", axis=1).values
    Y = df.Total.values
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    
    classifier = pipeline.Pipeline(
        [
            ("scaling", scl)
            ("pca", pca),
            ("rf", rf)
        ]
    )
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "pca__n_components": np.arange(5, 10), 
        "rf__n_estimators": [100, 200, 300, 400], 
        "rf__max_depth": [1, 3, 5, 7],
        "rf__criterion": ["gini", "entropy"],
    }
    
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1
        cv=5,
    )
    model.fix(X, Y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())