import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

class Plsr:
	def __init__(self, name_matrix, train_x, train_y, test_x, test_y, n_pcts, preprocessing, wavelength):
		self.name = name_matrix
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.n = train_x.shape[0]
		self.n_pcts = n_pcts
		self.preprocessing = preprocessing
		self.wavelength = wavelength
		self.plsr = PLSRegression(n_components = n_pcts, scale = True)

	# def _nm_to_index(self, indexes):
	# 	def _minus700_divide2(index):
	# 		return int((index - 700) / 2)
	# 	indexes = list(map(_minus700_divide2, list(map(int, indexes))))
	# 	return indexes

	#Used to compute the optimal # of principal component
	def _total_residual_variance(self, matrix):
		return np.sum((matrix - np.mean(matrix, axis = 0)) ** 2, axis = 0) / (matrix.shape[0])

	#Standardlize the data
	def _center_and_scale_data(self, x, y):

		self.x_mean = x.mean(axis = 0)
		x -= self.x_mean
		self.x_std = x.std(axis = 0, ddof = 1)
		self.x_std[self.x_std == 0.0] = 1.0

		self.y_mean = y.mean(axis = 0)
		y -= self.y_mean
		self.y_std = y.std(axis = 0, ddof = 1)
		self.y_std[self.y_std == 0.0] = 1.0

	def train(self):
		
		# self.coef = np.zeros(shape = (self.n_pcts, self.train_x[1], self.train_x[0]))
		self.pred_valid_y = np.zeros(shape = (self.train_y.shape[0], self.n_pcts))
		self.calibration_r2 = np.zeros((self.n_pcts, ))
		self.rmsecv = np.zeros((self.n_pcts, ))

		self.trv = np.zeros((self.n_pcts + 1, ))
		self.trv[0] = self._total_residual_variance(self.train_y)
		self.y_residual = np.zeros((self.train_y.shape[0], self.n_pcts))
		self.optimal_pc = 1

		#Cross validation (leave one out)
		for i in range(self.train_x.shape[0]):
			#Split calibration set and validation set
			calibration_idx = [x for x in range(self.train_x.shape[0]) if x != i]
			valid_x = self.train_x[i, :].reshape(1, -1)
			valid_y = self.train_y[i, :]
			calibration_x = self.train_x[calibration_idx, :]
			calibration_y = self.train_y[calibration_idx, :]
			
			#Center and scale data
			_calibration_x = np.copy(calibration_x)
			_calibration_y = np.copy(calibration_y)

			self._center_and_scale_data(_calibration_x, _calibration_y)

			#Fit and validate the data
			self.plsr.fit(calibration_x, calibration_y)

			#Validation
			for pct in range(self.n_pcts):
				_valid_x = np.copy(valid_x)
				coef = np.dot(self.plsr.x_rotations_[:, :pct + 1], \
						self.plsr.y_loadings_[:, :pct + 1].T) * self.y_std

				#For carspls
				# self.coef[pct] = coef

				_valid_x -= self.x_mean
				_valid_x /= self.x_std
				_valid_y = np.dot(_valid_x, coef) + self.y_mean
				self.pred_valid_y[i][pct] = _valid_y

				self.rmsecv[pct] += (valid_y - _valid_y) ** 2
		# plt.plot(np.arange(self.train_y.shape[0]), self.train_y[:, 0])
		# plt.savefig('./test1.png')
		# plt.clf()
		# plt.plot(np.arange(self.train_y.shape[0]), self.pred_valid_y[:, 5])
		# plt.savefig('./test2.png')
		self.plsr.fit(self.train_x, self.train_y)

		#Some indexes
		for pct in range(self.n_pcts):
			self.calibration_r2[pct] = r2_score(self.train_y, self.pred_valid_y[:, pct])
		
		self.y_residual = np.subtract(self.pred_valid_y, self.train_y)
		self.trv[1:] = self._total_residual_variance(self.y_residual)
		self.rmsecv /= self.train_x.shape[0]
		self.rmsecv **= 0.5

		self.PCFormula = [self.trv[0] * 0.01 * x + self.trv[x] for x in range(1, len(self.trv))]
		self.optimal_pc = self.PCFormula.index(min(self.PCFormula)) + 1

		return self.optimal_pc, self.calibration_r2, self.rmsecv

	def predict(self):

		#Number of testing samples
		m = self.test_y.shape[0]

		#self.pred_y need to be recorded
		self.pred_y = np.zeros(shape = (m, self.n_pcts))
		self.prediction_r2 = np.zeros((self.n_pcts, ))
		self.rmsep = np.zeros((self.n_pcts, ))
		self.bias = np.zeros((self.n_pcts, ))
		self.sep = np.zeros((self.n_pcts, ))

		self.test_x -= self.x_mean
		self.test_x /= self.x_std

		#Predict the testing set
		for pct in range(self.n_pcts):
			coef = np.dot(self.plsr.x_rotations_[:, :pct + 1], \
					self.plsr.y_loadings_[:, :pct + 1].T) * self.y_std

			_pred_y = np.dot(self.test_x, coef) + self.y_mean
			self.pred_y[:, pct] = _pred_y[:, 0]

			#Some indexes
			self.prediction_r2[pct] = r2_score(self.test_y, self.pred_y[:, pct])
		
		self.bias[:] = np.sum(np.subtract(self.pred_y, self.test_y), axis = 0) / m
		self.sep[:] = ((np.sum(np.subtract(np.subtract(self.pred_y, self.test_y), self.bias) ** 2, axis = 0))/ 
				(m - 1)) ** 0.5
		self.rmsep[:] = (np.sum(np.subtract(self.pred_y, self.test_y) ** 2, axis = 0) / m) ** 0.5
		self.prediction_std = np.array([np.std(self.test_y)] * self.n_pcts)
		self.prediction_rpd = self.prediction_std / self.sep

		return self.prediction_r2, self.rmsep, self.sep, self.prediction_std, self.prediction_rpd
	
	def saving_as_csv(self):
		path = './outputdata/0507agtron/'
		if not os.path.exists(path):
			os.makedirs(path)
		with open(f'{path}/{self.preprocessing}_{self.wavelength}.csv', 'w', newline = '') as f:
			csvwriter = csv.writer(f)

			title = ['Sample name']
			title.extend([str(x) for x in range(1, self.n_pcts + 1, 1)])
			title.append('Ref. Agtron')
			csvwriter.writerow(title)

			Agtron_result = np.hstack((np.hstack((self.name.reshape(self.name.shape[0], -1), 
				np.round(self.pred_y, 3))), self.test_y))
			csvwriter.writerows(Agtron_result)
			
			info = np.array(['Calibration R square', 'RMSECV', 'Prediction R square', 
				'RMSEP', 'SEP', 'STD', 'RPD']).reshape(7, -1)
			result = np.concatenate((np.round(self.calibration_r2, 3), np.round(self.rmsecv, 3), 
				np.round(self.prediction_r2, 3), np.round(self.rmsep, 3), np.round(self.sep, 3),
				np.round(self.prediction_std, 3), np.round(self.prediction_rpd, 3))).reshape(7, -1)
			result = np.hstack((info, result))
			
			csvwriter.writerow(['Optimal PC', str(self.optimal_pc)])
			csvwriter.writerows(result)


	# def saving_as_plots(self):

	# 	return None

# if __name__ == '__main__':
# 	a = ['700','704']
# 	b = np.random.random((10, 2))
# 	plsr = Plsr('i', b, b, b, b, 5, 'g', 'h')
# 	plsr.train()
# 	plsr.predict()