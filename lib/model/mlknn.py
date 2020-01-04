import numpy as np

from sklearn.neighbors import NearestNeighbors

class Mlknn(object):
	def __init__(self):
		pass
		
	def train(self, x, y, n_neighbors = 5, smooth = 1):
		
		self.n = n_neighbors
		self.y = y
		self.prior_1 = np.zeros(self.y.shape[1])
		self.prior_0 = np.zeros(self.y.shape[1])
		self.post_1 = np.zeros((self.y.shape[1], self.n + 1))
		self.post_0 = np.zeros((self.y.shape[1], self.n + 1))
		
		#Prior
		self.prior_1 = (np.sum(self.y, axis = 0) + smooth) / (self.y.shape[0] + 2 * smooth)
		self.prior_0 = 1.0 - self.prior_1

		#Posterior
		self.neigh = NearestNeighbors()
		self.neigh.fit(x)
		for i in range(self.y.shape[1]):
			c = np.zeros(self.n + 1)
			c_prime = np.zeros(self.n + 1)

			for j in range(x.shape[0]):
				idx_neigh = self.neigh.kneighbors(x[j].reshape(1, -1), n_neighbors = self.n + 1, return_distance = False)
				sigma = 0
				for k in range(1, self.n + 1):
					sigma = sigma + self.y[idx_neigh[0][k]][i]
				
				if self.y[j][i] == 1:
					c[sigma] += 1
				else:
					c_prime[sigma] += 1

			for j in range(self.n + 1):
				self.post_1[i][j] = (smooth + c[j]) / (smooth * (self.n + 1) + np.sum(np.array(c))) 
				self.post_0[i][j] = (smooth + c_prime[j]) / (smooth * (self.n + 1) + np.sum(np.array(c_prime))) 

	def predict(self, test_x):
		prediction = np.zeros((test_x.shape[0], self.y.shape[1]))
		for i in range(test_x.shape[0]):
			idx_neigh = self.neigh.kneighbors(test_x[i].reshape(1, -1), n_neighbors = self.n + 1, return_distance = False)
			for j in range(self.y.shape[1]):
				sigma = 0
				for k in range(1, self.n + 1):
					sigma = sigma + self.y[idx_neigh[0][k]][j]
				if self.prior_1[j] * self.post_1[j][sigma] > self.prior_0[j] * self.post_0[j][sigma]:
					prediction[i][j] = 1
				else:
					prediction[i][j] = 0

		return prediction.astype(np.int)