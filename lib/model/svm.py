import numpy as np

from sklearn.svm import SVC

class Svm(object):
	def __init__(self):
		pass

	def train(self, x, y, kernel, C = 1):
		self.model = []
		for i in range(y.shape[1]):
			self.model.append(SVC(C = float(C), gamma = 'auto', kernel = kernel))
			self.model[i].fit(x, y[:,i])
		# print('Training is done!')

	def predict(self, x):
		prediction = np.zeros((x.shape[0], len(self.model)))
		for i in range(len(self.model)):
			prediction[:, i] = self.model[i].predict(x)
		# print('Predicting is done!')
		return prediction