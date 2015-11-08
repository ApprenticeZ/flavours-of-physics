# blending

import numpy as np
import pandas as pd
from sklearn import linear_model
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# load training and test set
print("Load the training/test data using pandas")
train = pd.read_csv("../../data/training.csv")
test  = pd.read_csv("../../data/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5]) # names of features
print('feature names', features)

# randomly split the training into a training set and a hold-out set
X_train, X_hold = train_test_split(train, test_size=0.1, random_state=1)

# train a base-line LR model on the subtrain set
logreg = linear_model.LogisticRegression()
logreg.fit(X_train[features], X_train["signal"])

# train a base-line RF model on the subtrain set
rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(X_train[features], X_train["signal"])

# train a blending LR model on the hold-out set
lr_train = logreg.predict_proba(X_hold[features])[:,1]
rf_train = rf.predict_proba(X_hold[features])[:,1]
blend_train = pd.DataFrame({"lr_pred": lr_train, "rf_pred": rf_train})
blend_LR = linear_model.LogisticRegression()
blend_LR.fit(blend_train, X_hold["signal"])

# make prediction on the test set
# predict with the base-line LR and RF model respectively
lr_test = logreg.predict_proba(test[features])[:,1]
rf_test = rf.predict_proba(test[features])[:,1]
blend_test = pd.DataFrame({"lr_pred": lr_test, "rf_pred": rf_test})
# predict with the blending LR model
test_probs = blend_LR.predict_proba(blend_test)[:,1]
submission = pd.DataFrame({"id": test['id'], "prediction": test_probs})
submission.to_csv("../../submission/blend_base.csv", index=False) # 