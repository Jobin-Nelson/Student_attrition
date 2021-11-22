import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from functools import partial
import optuna


def optimize(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators",10,300,)
    learning_rate = trial.suggest_uniform("learning_rate",0.05, 2.5)
    loss = trial.suggest_categorical("loss",["deviance","exponential"])
    max_depth = trial.suggest_int("max_depth",1,7)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss= loss,
        max_depth=max_depth
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        n_jobs=-1,
    )

    accuracy = score.mean()
    return -1*accuracy

if __name__ == "__main__":
    df = pd.read_csv("..\\cleaned_data\\Cleaned_data.csv")
    X = df.drop("Target",axis=1).values
    y = df["Target"].values

    optimization_func = partial(optimize, X=X, y=y)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_func,n_trials=50)
    print(study.best_trial)