__author__ = 'zhang'

import numpy as np
import pandas as pd
from sklearn.svm import SVC

print("Load the training/test data using pandas")
target = pd.read_csv('../../data/training.csv')
train = pd.read_csv("../../data/ntrain.csv")
test  = pd.read_csv("../../data/ntest.csv")
testId = pd.read_csv('../../data/test.csv')['id']

# print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns) # names of features
print(features)

print("Train a SVM")
svm = SVC(C=32.0,kernel='rbf',gamma=0.0078125)
svm.fit(train[features], target["signal"])

# predict with test data
print('prediction')
test_probs = svm.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": testId, "prediction": test_probs})
submission.to_csv("../../submission/svm.csv", index=False) 