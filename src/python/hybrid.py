# a hybrid model
# use gradient boost tree to transform features 
# and train a linear regression model for classification

import numpy as np
import pandas as pd
from sklearn import linear_model
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

print("Load the training/test data using pandas")
train = pd.read_csv("../../data/training.csv")
test  = pd.read_csv("../../data/test.csv")

# xgboost only accepts alphanumeric feature names
# remove '_' in feature names
replaceNonAlpha = lambda s:s.replace('_','')
train.rename(columns=replaceNonAlpha,inplace=True)
test.rename(columns=replaceNonAlpha,inplace=True)

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5]) # names of features
print('feature names', features)

# print("Load a XGBoost model")
# bst = xgb.Booster(model_file='../../model/base.model')

# randomly split the training into a training set and a hold-out set
X_train, X_hold = train_test_split(train, test_size=0.5, random_state=1)

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=10
bst = xgb.train(params, xgb.DMatrix(X_train[features], X_train["signal"]), num_trees)
bst.save_model('../../model/10_feature_transform.model')

# print('Show feature importance')
# xgb.plot_importance(bst)
# plt.show()

# train a one-hot encoder
ohe = OneHotEncoder()
dtrain = xgb.DMatrix(X_train[features])
trainLeaf = bst.predict(dtrain, pred_leaf=True) # predict with all trees
ohe.fit(trainLeaf) 

# transform training features
ohe_feature = ohe.transform(bst.predict(xgb.DMatrix(X_hold[features]), pred_leaf=True))

# build a LR model
print("Train a Logistic Regression model")
logreg = linear_model.LogisticRegression()
logreg.fit(ohe_feature, X_hold["signal"])

# transform test features
dtest = xgb.DMatrix(test[features])
testLeaf = bst.predict(dtest, pred_leaf=True)
testLeaf = ohe.transform(testLeaf)

# predict with LR model
test_probs = logreg.predict_proba(testLeaf)[:,1]
submission = pd.DataFrame({"id": test['id'], "prediction": test_probs})
submission.to_csv("../../submission/hybrid_base.csv", index=False) # 

# hybrid_rf
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(trainLeaf, train["signal"])
test_probs = rf.predict_proba(testLeaf)[:,1] 
submission = pd.DataFrame({"id": test['id'], "prediction": test_probs})
submission.to_csv("../../submission/hybrid_rf_base.csv", index=False) # 