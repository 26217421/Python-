import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

data = pd.read_csv("yuan.csv", header=None)
# data=np.array(data)
X = data[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values
y = data[[1]].values
y = np.ravel(y)
print(y)
lda = LinearDiscriminantAnalysis(n_components=2)

X_r2 = lda.fit(X, y).transform(X)
print(X_r2)
target_names = data[1].unique()
print()
plt.figure()

for c, i, target_name in zip(['k', 'b', 'r', 'y', 'g', 'c', 'm'], target_names,
                             target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of the dataset')

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.tight_layout()

plt.legend(loc='upper right')
plt.show()
