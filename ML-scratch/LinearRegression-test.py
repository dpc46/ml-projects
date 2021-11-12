import numpy as np
from numpy.lib.function_base import meshgrid
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s=30)
# plt.show()

# print(X_train.shape)
# print(y_train.shape)

from LinearRegression import LinearRegression

regressor = LinearRegression(n_iters=10000)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mean_sqaured_error(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mean_sqaured_error(y_test, predicted)
print(mse_value)

best_fit_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, best_fit_line, color="k", linewidth=2, label="Prediction")
plt.show()

