import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['yellow', 'red', 'purple'])

from KNN import KNN

irisDataSet = datasets.load_iris()
X, y = irisDataSet.data, irisDataSet.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()


clf = KNN(k=3)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print(prediction)

acc = np.sum(prediction==y_test) / len(y_test)
print(acc)