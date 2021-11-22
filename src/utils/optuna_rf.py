import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from functools import partial
import optuna

def optimize(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators",100,200)
    criterion = trial.suggest_categorical("criterion",["gini","entropy"])
    max_depth = trial.suggest_int("max_depth",2,10)

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth)

    kf = StratifiedKFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=X,y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = X[train_idx]
        ytrain = y[train_idx]

        xtest = X[test_idx]
        ytest = y[test_idx]

        rf.fit(xtrain, ytrain)
        preds = rf.predict(xtest)
        fold_acc = accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0*np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("..\\cleaned_data\\Cleaned_data.csv")
    X = df.drop('Target', axis=1).values
    y = df['Target'].values

    optimization_func = partial(optimize, X=X, y=y)

    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_func, n_trials = 50)