import csv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression

class Plsr:

	def __init__(self, train_x, train_y, test_x, test_y, n_pcts, preprocessing, wavelength):
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.n = train_x.shape[0]
		self.n_pcts = n_pcts
		self.preprocessing = preprocessing
		self.wavelength = wavelength
		self.info = ['calibration r square', 'calibration rmsecv', 
			'prediction r square', 'prediction rmsep', 'prediction sep', 
			'prediction std', 'prediction rpd']

		self.cal_r_square = []
		self.rmsecv_history = []
		self.optimal_pc = 1

		self.pre_r_square = []
		self.rmsep_history = []
		self.pre_sep = []
		self.pre_std = []
		self.pre_rpd = []
		self.pre_y = []
		
	def main(self):
		for pc in tqdm(range(1, self.n_pcts + 1, 1)):
			cal_r = 0.0
			rmsecv = 0.0

			pre_r = 0.0
			rmsep = 0.0
			sep = 0.0
			std = 0.0
			rpd = 0.0

			bias = 0.0

			plsr = PLSRegression(n_components = pc)

			#LOO cross validation
			for i in tqdm(range(self.train_x.shape[0])):
				idx = [x for x in range(self.train_x.shape[0]) if x != i]
				valid_x = self.train_x[i, :].reshape(1, -1)
				valid_y = self.train_y[i, :]
				calibration_x = self.train_x[idx, :]
				calibration_y = self.train_y[idx, :]

				plsr.fit(calibration_x, calibration_y)
				cal_r += plsr.score(calibration_x, calibration_y)
				rmsecv += np.sum((plsr.predict(valid_x) - valid_y) ** 2)

			plsr.fit(self.train_x, self.train_y)

			cal_r /= self.train_x.shape[0]
			rmsecv /= self.train_x.shape[0]
			rmsecv = rmsecv ** 0.5

			pre_y = plsr.predict(self.test_x)
			bias = np.sum(pre_y - self.test_y) / self.test_x.shape[0]

			pre_r = plsr.score(self.test_x, self.test_y)
			rmsep = ((np.sum((pre_y - self.test_y) ** 2)) / self.test_x.shape[0]) ** 0.5
			sep = ((np.sum((pre_y - self.test_y - bias) ** 2))/(self.test_x.shape[0] - 1)) ** 0.5
			std = np.std(self.test_y)
			rpd = std/sep

			self.cal_r_square.append(cal_r)
			self.rmsecv_history.append(rmsecv)

			self.pre_r_square.append(pre_r)
			self.rmsep_history.append(rmsep)
			self.pre_sep.append(sep)
			self.pre_std.append(std)
			self.pre_rpd.append(rpd)

			self.pre_y.append(pre_y)

		self.pre_y.append(self.test_y)
		self.optimal_pc = self.rmsep_history.index(min(self.rmsep_history)) + 1

		print(f'calibration r square: {self.cal_r_square}')
		print(f'calibration rmsecv: {self.rmsecv_history}')
		print(f'calibration factor: {self.optimal_pc}')
		print(f'prediction r square: {self.pre_r_square}')
		print(f'prediction rmsep: {self.rmsep_history}')
		print(f'prediction sep: {self.pre_sep}')
		print(f'prediction std: {self.pre_std}')
		print(f'prediction rpd: {self.pre_rpd}')

		cp = np.arange(1, 11, 1)
		plt.plot(cp, self.rmsecv_history, label = 'RMSECV')
		plt.plot(cp, self.rmsep_history, label = 'RMSEP')
		plt.legend(loc = 'upper right')
		plt.savefig(f'abcd_{self.preprocessing}_{self.wavelength}.png')
		plt.clf()


		with open(f'pre_{self.preprocessing}_{self.wavelength}.csv', 'w', newline = '') as f:
			s = csv.writer(f)
			a = [str(x) for x in range(1, self.n_pcts + 1, 1)]
			a.append('Ground Truth')
			s.writerow(a)
			for i in range(self.test_y.shape[0]):
				s.writerow([self.pre_y[j][i][0] for j in range(len(self.pre_y))])
			
			s.writerow(str(self.optimal_pc))
			s.writerow([self.info[0], [self.cal_r_square[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[1], [self.rmsecv_history[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[2], [self.pre_r_square[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[3], [self.rmsep_history[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[4], [self.pre_sep[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[5], [self.pre_std[i] for i in range(self.n_pcts)]])
			s.writerow([self.info[6], [self.pre_rpd[i] for i in range(self.n_pcts)]])

		return None