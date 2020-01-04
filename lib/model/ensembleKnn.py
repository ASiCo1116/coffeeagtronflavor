import os
import csv
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

def train_test_split(x, y, start, ratio):
	step = int(1 / ratio) - 1
	test_idx = list(range(start, x.shape[0], step))
	train_x = x[[i for i in range(x.shape[0]) if i not in test_idx], :]
	train_y = y[[i for i in range(x.shape[0]) if i not in test_idx], :]
	test_x = x[test_idx, :]
	test_y = y[test_idx, :]

	return train_x, train_y, test_x, test_y

x = np.load('../../outputdata/nir_a1_b1_b33_700_2498.npy', allow_pickle = True)
y = np.load('../../outputdata/flavor_a1_b1_b33.npy', allow_pickle = True)
nan_idx = ~pd.isnull(y).any(axis = 1)
x = x[nan_idx]
y = y[nan_idx]

train_x, train_y, test_x, test_y = train_test_split(x, y, 6, 0.125)
test_name_x = test_x[:, :2]
train_x = train_x[:, 2:].astype(np.float64)
train_y = train_y[:, 2:].astype(np.int)
test_x = test_x[:, 2:].astype(np.float64)
test_y = test_y[:, 2:].astype(np.int)

print(f'train_x :{train_x.shape}, train_y : {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}')

for i in range(100, 200):
	acc = 0.0
	rec = 0.0
	true = 0.0

	floral = KNeighborsClassifier(n_neighbors = i)
	fruity = KNeighborsClassifier(n_neighbors = i)
	sf = KNeighborsClassifier(n_neighbors = i)
	veg = KNeighborsClassifier(n_neighbors = i)
	other = KNeighborsClassifier(n_neighbors = i)
	roasted = KNeighborsClassifier(n_neighbors = i)
	spices = KNeighborsClassifier(n_neighbors = i)
	nc = KNeighborsClassifier(n_neighbors = i)
	sweet = KNeighborsClassifier(n_neighbors = i)

	floral.fit(train_x, train_y[:, 0])
	fruity.fit(train_x, train_y[:, 6])
	sf.fit(train_x, train_y[:, 9])
	veg.fit(train_x, train_y[:, 10])
	other.fit(train_x, train_y[:, 11])
	roasted.fit(train_x, train_y[:, 12])
	spices.fit(train_x, train_y[:, 13])
	nc.fit(train_x, train_y[:, 14])
	sweet.fit(train_x, train_y[:, 15])

	with open(f'../../outputdata/knn_flavor/knn_flavor_classification_n{i}.csv', 'w', newline = '') as f:
		c = csv.writer(f)
		c.writerow(['name', 'number', 'floral', 'fruity', 'sf', 'veg', 'other', 'roasted', 'spices', 'nc', 'sweet'])
		for j in range(test_x.shape[0]):
			a = []
			a.append(test_name_x[j, 0])
			a.append(test_name_x[j, 1])
			a.append(floral.predict(test_x[j].reshape(1, -1)))
			a.append(fruity.predict(test_x[j].reshape(1, -1)))
			a.append(sf.predict(test_x[j].reshape(1, -1)))
			a.append(veg.predict(test_x[j].reshape(1, -1)))
			a.append(other.predict(test_x[j].reshape(1, -1)))
			a.append(roasted.predict(test_x[j].reshape(1, -1)))
			a.append(spices.predict(test_x[j].reshape(1, -1)))
			a.append(nc.predict(test_x[j].reshape(1, -1)))
			a.append(sweet.predict(test_x[j].reshape(1, -1)))
			c.writerow(a)

			b = ['.', '.']
			b.append(test_y[j, 0])
			b.append(test_y[j, 6])
			b.append(test_y[j, 9])
			b.append(test_y[j, 10])
			b.append(test_y[j, 11])
			b.append(test_y[j, 12])
			b.append(test_y[j, 13])
			b.append(test_y[j, 14])
			b.append(test_y[j, 15])
			c.writerow(b)

			for k in range(2, len(a)):
				if a[k] == b[k]:
					acc += 1.0
				if b[k] == 1 and a[k] == 1:
					rec += 1.0
				if b[k] == 1:
					true += 1.0

		acc /= (9 * test_x.shape[0])
		rec /= true

		print(f'neighbors: {i}, {acc * 100} %, {rec * 100} %')



