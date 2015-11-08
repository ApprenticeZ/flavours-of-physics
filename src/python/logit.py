__author__ = 'zhang'

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

print("Load the training/test data using pandas")
train = pd.read_csv("../../data/training.csv")
test  = pd.read_csv("../../data/test.csv")

# print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5]) # names of features

# param grid
param_grid = [{'C': [1,10,100,1000]
                            ,'class_weight':['auto',{0:0.2,1:0.8},{0:0.4,1:0.6},{0:0.5,1:0.5}]}]

def _score(model, X, labels):
    pred = model.predict_proba(X)[:,1]
    return evaluation.roc_auc_truncated(labels, pred)

print("Train a Logistic Regression model")
logreg = linear_model.LogisticRegression(random_state=1)
clf = GridSearchCV(logreg, param_grid, scoring=_score, cv=10)
clf.fit(train[features], train["signal"])
print(clf.best_params_)

# predict with test data
test_probs = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test['id'], "prediction": test_probs})
submission.to_csv("../../submission/logreg_tune.csv", index=False)