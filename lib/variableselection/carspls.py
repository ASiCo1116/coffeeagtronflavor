import os
import csv
import numpy as np
import pandas as pd

from lib.utils.package import read_data
from lib.utils.wave_preprocess import raw, msc, sg1222_msc, msc_sg1222, wave_select

from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

def random_sample(ratio, sample_size):
	choose = list(np.random.randint(sample_size, size = int(ratio * sample_size)))
	return choose

class Carspls():
	def __init__(self, n_components, num_run, sample_ratio):
		
		self.n_components = n_components
		self.plsr = PLSRegression(n_components = self.n_components)

		self.name = name_matrix
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.n = train_x.shape[0]
		self.n_pcts = n_pcts
		self.preprocessing = preprocessing
		self.wavelength = wavelength

		self.num_run = num_run #num of sampling runs
		self.ratio = sample_ratio
		self.variable_set = list(range(self.p))

	def train(self, x, y, sample, wave, run, replace = True):
		self.wave = wave
		self.p = x.shape[1]
		self.plsr.fit(x[sample, self.wave], y[sample])
		
		b = plsr.coef_
		b = abs(b)
		w = b / np.sum(b)
		w = w.squeeze().tolist()
		self.wave = sorted(list(dict.fromkeys(self.ARS(self.wave, int(self.retained_ratio(run)), w, replace))))
		
		self.rmsecv = 0.0
		self.cal_r2 = 0.0
		all_pred_y = np.zeros(shape = (len(sample), 1))
		#Leave-one-out
		for i in range(len(sample)):
			#Split calibration set and validation set
			calibration_idx = [idx for idx in range(len(sample)) if idx != i]
			valid_x = x[i, self.wave].reshape(1, -1)
			valid_y = y[i, :]
			calibration_x = x[calibration_idx, self.wave]
			calibration_y = y[calibration_idx, :]

			#Validate the data
			pred_y = self.plsr.predict(valid_x)
			all_pred_y[i] = pred_y
			self.rmsecv += (valid_y - pred_y) ** 2

		self.rmsecv /= len(sample)
		self.rmsecv **= 0.5
		# trv = self.total_residual_variance(np.subtract(all_pred_y, y[sample]))
		self.cal_r2 = r2_score(y[sample], all_pred_y)
		
		# return trv

	def predict(self, x, y):
		self.y = y
		m = self.y.shape[0]

		self.prediction = self.plsr.predict(x[:, self.wave])

		bias = np.sum(np.subtract(self.prediction, self.y), axis = 0) / m
		self.sep = ((np.sum(np.subtract(np.subtract(self.prediction, self.y), bias) ** 2, axis = 0))/ 
				(m - 1)) ** 0.5
		self.rmsep = (np.sum(np.subtract(self.prediction, self.y) ** 2, axis = 0) / m) ** 0.5
		self.std = np.std(self.y, axis = 0)
		self.rpd = self.std / self.sep
		self.pre_r2 = r2_score(y, self.prediction)

	def retained_ratio(self, i):
		if i == 1:
			return 1
		elif i == self.num_run:
			return 2/self.p
		else:
			return (self.p/2) ** (1 / (self.num_run - 1)) * math.exp(-math.log(self.p / 2) / (self.num_run - 1) * i)
	
	# def plsr(self, x, y):
	# 	plsr = PLSRegression(self.n_pcts)
	# 	plsr.fit(x, y)
	# 	b = plsr.coef_ #coef of wavelength
	# 	b = abs(b)
	# 	w = b / np.sum(b)
	# 	return w.squeeze().tolist()

	# def cross_validation(self, x, y):
	# 	plsr = PLSRegression(self.n_pcts)
	# 	RMSECV = 0.0
	# 	R_square = 0.0
	# 	for j in range(x.shape[0]):
	# 		test_x = x[j, :].reshape(1, -1)
	# 		test_y = y[j]
	# 		idx = np.array([num for num in range(x.shape[0]) if num != j])
	# 		train_x = x[idx, :]
	# 		train_y = y[idx]
	# 		plsr.fit(train_x, train_y)
	# 		RMSECV += np.sum((test_y - plsr.predict(test_x)) ** 2)
	# 		R_square += plsr.score(train_x, train_y)
	# 	return (RMSECV / x.shape[0]) ** 0.5

	def ARS(self, wave, num_retained, propability, replace = True):
		return np.random.choice(wave, replace = replace, size = num_retained, p = propability)

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

if __name__ == '__main__':
	main()