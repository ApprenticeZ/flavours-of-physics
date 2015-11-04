__author__ = 'zhang'

import numpy as np
import pandas as pd
from sklearn import linear_model

print("Load the training/test data using pandas")
target = pd.read_csv('../../data/training.csv')
train = pd.read_csv("../../data/ntrain.csv")
test  = pd.read_csv("../../data/ntest.csv")
testId = pd.read_csv('../../data/test.csv')['id']

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5]) # names of features

print("Train a Logistic Regression model")
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train[features], target["signal"])

# predict with test data
test_probs = logreg.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": testId, "prediction": test_probs})
submission.to_csv("../../submission/logreg_featNorm.csv", index=False) # 0.971263