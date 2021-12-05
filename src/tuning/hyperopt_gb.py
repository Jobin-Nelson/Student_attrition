import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, X, y):
    model = GradientBoostingClassifier(**params)
    kf = StratifiedKFold(n_splits=5)

    accuracy = []
    for idx in kf.split(X,y):
        train_idx, test_idx = idx[0], idx[1]

        xtrain = X[train_idx]
        ytrain = y[train_idx]

        xtest = X[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        ypreds = model.predict(xtest)

        fold_acc = accuracy_score(ytest, ypreds)
        accuracy.append(fold_acc)

    return -1*np.mean(accuracy)

if __name__ =="__main__":
    df = pd.read_csv("..\\cleaned_data\\Cleaned_data.csv")
    X = df.drop("Target", axis = 1).values
    y = df["Target"].values

    optmizing_func = partial(optimize, X=X, y=y)
    trials = Trials()
    
    space = {
        "n_estimators":scope.int(hp.quniform("n_estimators",10,250,1)),
        "learning_rate":hp.uniform("learning_rate",0.05,2.5),
        "loss":hp.choice("loss",["deviance","exponential"]),
        "max_depth":scope.int(hp.quniform("max_depth",1,7,1))
    }

    result = fmin(
        fn = optmizing_func,
        trials=trials,
        max_evals=50,
        algo=tpe.suggest,
        space=space
    )

    print(result)