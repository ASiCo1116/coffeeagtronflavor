import csv
import math
import numpy as np

from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression

class Carspls:
	def __init__(self, x, y, n, monte_carlo_ratio, n_cpts, preprocessing):
		self.x = x
		self.y = y
		self.compts = n_cpts
		self.num_run = n #num of sampling runs
		self.m = x.shape[0]	#num of data
		self.p = x.shape[1]	#total num of wavelength(variables)
		self.ratio = monte_carlo_ratio
		self.variable_set = list(range(self.p))
		self.preprocessing = preprocessing

	def retained_ratio(self, i):
		if i == 1:
			return 1
		elif i == self.num_run:
			return 2/self.p
		else:
			return (self.p/2) ** (1 / (self.num_run - 1)) * math.exp(-math.log(self.p / 2) / (self.num_run - 1) * i)
	
	def plsr(self, x, y):
		plsr = PLSRegression(self.compts)
		plsr.fit(x, y)
		b = plsr.coef_ #coef of wavelength
		b = abs(b)
		w = b / np.sum(b)
		return w.squeeze().tolist()

	def cross_validation(self, x, y):
		plsr = PLSRegression(self.compts)
		RMSECV = 0.0
		R_square = 0.0
		for j in range(x.shape[0]):
			test_x = x[j, :].reshape(1, -1)
			test_y = y[j]
			idx = np.array([num for num in range(x.shape[0]) if num != j])
			train_x = x[idx, :]
			train_y = y[idx]
			plsr.fit(train_x, train_y)
			RMSECV += np.sum((test_y - plsr.predict(test_x)) ** 2)
			R_square += plsr.score(train_x, train_y)
		return (RMSECV / x.shape[0]) ** 0.5

	def random_sample(self, val_sel):
		choose = list(np.random.randint(self.m, size = int(self.ratio * self.m)))
		x = self.x[choose, :]
		y = self.y[choose, :]
		return x[:, val_sel], y

	def ARS(self, val_sel, num_retained, propability):
		return np.random.choice(val_sel, replace = True, size = num_retained, p = propability)

	def main(self):
		with open(f'output_{self.preprocessing}.csv', 'w', newline = '') as csvfile:
			f = csv.writer(csvfile)
			f.writerow(['Subset', 'Wavelength', '# of wavelength', 'RMSECV'])
			for i in range(1, self.num_run + 1):
				print(f'No. {i}')
				x, y = self.random_sample(self.variable_set)
				weight = self.plsr(x, y)
				ri = int(self.p * self.retained_ratio(i))
				rmsecv = self.cross_validation(x, y)
				self.variable_set = sorted(list(dict.fromkeys(self.ARS(self.variable_set, ri, weight))))
				f.writerow([str(i), [str(w * 2 + 700) for w in self.variable_set], str(len(self.variable_set)), str(rmsecv)])
		return None