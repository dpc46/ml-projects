import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# splitting data into targets and features, training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=1234)

#  # printing the shape of the x data, and an individual vector
# print(f"Number of rows by number of features:\t{X_train.shape}\n")
# print(f"Example x vector:\t{X_train[0]}\n")

# # printing the shape of the target data 
# print(f"Number of rows by target size:\t{y_train.shape}\n")
# print(f"Unique values in the target data "
# f"{set(y_train.tolist())}")

# # printing only the first two features and the targets for all data
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=20)
# plt.show()


from knearest import KNN
neighbours = 3
classifer = KNN(k=neighbours)
# note for k-nearest neighbours we do not really 'fit', rather simply import the data
classifer.fit(X_train, y_train)
predictions = classifer.predict(X_test)

# accuracy measurement
acc = np.sum(predictions == y_test) / len(y_test)
result = f"Accuracy is {acc:.0%} using k = {neighbours} nearest neighbours"
print("\n" + "{:*^60}".format(result) + "\n")
