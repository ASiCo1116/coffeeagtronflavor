import numpy as np

from sklearn.neighbors import KNeighborsClassifier

class Knn(object):
	def __init__(self):
		pass

	def train(self, x, y, n_neighbors = 5, weights = 'uniforms'):
		self.model = []
		for i in range(y.shape[1]):
			self.model.append(KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights))
			self.model[i].fit(x, y[:,i])
		# print('Training is done!')

	def predict(self, x):
		prediction = np.zeros((x.shape[0], len(self.model)))
		for i in range(len(self.model)):
			prediction[:, i] = self.model[i].predict(x)
		# print('Predicting is done!')
		return prediction

