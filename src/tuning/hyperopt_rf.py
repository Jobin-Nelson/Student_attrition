import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from functools import partial
from hyperopt import hp, fmin, Trials, tpe
from hyperopt.pyll.base import scope

def optimize(params, X, y):
    model = RandomForestClassifier(**params)
    kf = StratifiedKFold(n_splits=5)

    recalls = []
    for idx in kf.split(X, y):
        train_idx, test_idx = idx[0], idx[1]

        xtrain = X[train_idx]
        ytrain = y[train_idx]

        xtest = X[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        ypreds = model.predict(xtest)
        fold_rec = recall_score(ytest, ypreds)
        recalls.append(fold_rec)

    return -1*np.mean(recalls)

if __name__ == "__main__":
    df = pd.read_csv("..\\cleaned_data\\Cleaned_data.csv")
    X = df.drop("Target", axis = 1).values
    y = df["Target"].values

    param_space = {
        "n_estimators":scope.int(hp.quniform("n_estimators",100,300,1)),
        "criterion": hp.choice("criterion",["gini","entropy"]),
        "max_depth":scope.int(hp.quniform("max_depth",1,7,1)),
    }

    optimization_func = partial(optimize, X=X, y=y)

    result = fmin(fn=optimization_func,
                  max_evals=15,
                  algo=tpe.suggest,
                  trials=Trials(),
                  space=param_space)
    print(result)