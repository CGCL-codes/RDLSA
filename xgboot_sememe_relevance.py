import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def loadData():
    # Load the data from xgcorpus.txt
    df1 = pd.read_csv("./data/xgcorpus.txt", delimiter="\t", header=None)
    df2 = pd.read_csv("./data/negXgcorpus.txt", delimiter="\t", header=None)
    df2.iloc[:, 3] = 0
    data = pd.concat([df1, df2])

    X = data.iloc[:, :3].astype(float)  # Features a, b, c
    # X.fillna(0, inplace=True)
    y = data.iloc[:, 3].astype(float)  # Target variable
    print(X.head(), y[:5])
    return X, y

X, y = loadData()

print('ready for data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# create XGBoost model and train on the training set
model = xgb.XGBRegressor()
print('initiation for model')
model.fit(X_train, y_train)

# get feature importance scores
importance = model.feature_importances_

# print feature importance scores
for i,v in enumerate(importance):
    print('Feature %d: %.5f' % (i+1,v))

# make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred)