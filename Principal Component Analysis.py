# Write your codes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_defect = pd.read_csv('wine.csv')

df_defect.shape

df_defect.describe()

X = df_defect.drop(['Wine'], axis=1)
y = df_defect['Wine']

print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

