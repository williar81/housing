from sklearn import datasets
import numpy as np

data_set = datasets.load_boston()
data, target = data_set.data, data_set.target

np.delete(data, [1, 10], axis=1)
new_data = data[:, :-1]

print(new_data.shape)
# print(target)
