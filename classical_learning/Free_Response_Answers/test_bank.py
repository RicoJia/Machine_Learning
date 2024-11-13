#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

data = pd.read_csv("./bank-additional/bank-additional-full.csv", sep=";")
# trucated columns set to 500 so we can see all columns
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 50)
# print(data)

data["no_previous_contact"] = np.where(data["pdays"] == 999, 1, 0)
data["not_working"] = np.where(
    np.in1d(data["job"], ["student", "retired", "unemployed"]), 1, 0
)

model_data = pd.get_dummies(data)
model_data = model_data.drop(
    [
        "y_no",
        "duration",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ],
    axis=1,
)
print(model_data)

train_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729), [int(0.9 * len(model_data))]
)
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, 59]

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, 59]

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# print(train_data.shape, test_data.shape)

# !pip install xgboost

import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split

X, val_X, y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=2022, stratify=train_y
)
# Convert datasets into DMatrix format for XGBoost
xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_x)

# XGBoost parameters
params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "gamma": 0.1,  # tree trimming parameter?
    "max_depth": 8,  # Maximum tree depth
    "alpha": 0,  # L1 regularization
    "lambda": 10,  # L2 regularization
    "subsample": 0.7,  # Sample ratio of training instances
    "colsample_bytree": 0.5,  # Ratio of columns when constructing each tree
    "min_child_weight": 3,  # Minimum sum of instance weight needed in a child
    "silent": 0,
    "eta": 0.03,  # Learning rate
    "seed": 1000,  # Random seed
    "nthread": -1,  # CPU threads
    "missing": None,  # Default missing value parameter
    "scale_pos_weight": (
        np.sum(y == 0) / np.sum(y == 1)
    ),  # Balancing for imbalanced datasets
}

plst = list(params.items())
num_rounds = 500  # Maximum number of boosting iterations
watchlist = [(xgb_train, "train"), (xgb_val, "val")]

# Train XGBoost model with early stopping
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
model.save_model("xgb.model")  # Save the trained model as xgb.model

preds = model.predict(xgb_test)

# Apply a threshold to convert probabilities to binary outcomes
threshold = 0.5
ypred = np.where(preds > 0.5, 1, 0)

# Print evaluation metrics
print("AUC: %.4f" % metrics.roc_auc_score(test_y, ypred))
print("ACC: %.4f" % metrics.accuracy_score(test_y, ypred))
print("Recall: %.4f" % metrics.recall_score(test_y, ypred))
print("F1-score: %.4f" % metrics.f1_score(test_y, ypred))
print("Precision: %.4f" % metrics.precision_score(test_y, ypred))

# Print the confusion matrix
print(metrics.confusion_matrix(test_y, ypred))
