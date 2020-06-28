import os
import csv
import numpy as np
import pandas as pd

from lib.utils.package import read_data
from lib.utils.wave_preprocess import raw, msc, sg1222_msc, msc_sg1222, wave_select

from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

class Plsr:
	def __init__(self, n_components):
		self.n_components = n_components
		self.plsr = PLSRegression(n_components = self.n_components)

	def optimal_pc(self, all_trv):
		formula = [all_trv[0] * 0.01 * x + all_trv[x] for x in range(1, all_trv.shape[0])]
		optimal_pc = formula.index(min(formula)) + 1
		print('Optimal :', optimal_pc)

	#Used to compute the optimal # of principal component
	def total_residual_variance(self, matrix):
		return np.sum((matrix - np.mean(matrix, axis = 0)) ** 2, axis = 0) / (matrix.shape[0])

	def train(self, x, y):
		self.rmsecv = 0.0
		self.cal_r2 = 0.0
		all_pred_y = np.zeros(shape = y.shape)
		#Leave-one-out
		for i in range(x.shape[0]):
			#Split calibration set and validation set
			calibration_idx = [idx for idx in range(x.shape[0]) if idx != i]
			valid_x = x[i, :].reshape(1, -1)
			valid_y = y[i, :]
			calibration_x = x[calibration_idx, :]
			calibration_y = y[calibration_idx, :]

			#Fit and validate the data
			self.plsr.fit(calibration_x, calibration_y)
			pred_y = self.plsr.predict(valid_x)
			all_pred_y[i] = pred_y
			self.rmsecv += (valid_y - pred_y) ** 2

		self.rmsecv /= x.shape[0]
		self.rmsecv **= 0.5
		trv = self.total_residual_variance(np.subtract(all_pred_y, y))
		self.cal_r2 = r2_score(y, all_pred_y)
		self.plsr.fit(x, y)
		return trv

	def predict(self, x, y):
		self.y = y
		m = self.y.shape[0]

		self.prediction = self.plsr.predict(x)

		bias = np.sum(np.subtract(self.prediction, self.y), axis = 0) / m
		self.sep = ((np.sum(np.subtract(np.subtract(self.prediction, self.y), bias) ** 2, axis = 0))/ 
				(m - 1)) ** 0.5
		self.rmsep = (np.sum(np.subtract(self.prediction, self.y) ** 2, axis = 0) / m) ** 0.5
		self.std = np.std(self.y, axis = 0)
		self.rpd = self.std / self.sep
		self.pre_r2 = r2_score(y, self.prediction)

	def save_to_csv(self, name, preprocessing, wavelength):
		path = './outputdata/0516agtron/'
		if not os.path.exists(path):
			os.makedirs(path)
		with open(f'{path}/plsr_{self.n_components}_{preprocessing}_{wavelength}.csv', 'w', newline = '') as f:
			csvwriter = csv.writer(f)

			title = ['Sample name', 'Pred. Agtron', 'Ref. Agtron']
			# title.extend([str(x) for x in range(1, self.n_pcts + 1, 1)])
			csvwriter.writerow(title)

			Agtron_result = np.hstack((np.hstack((name.reshape(name.shape[0], -1), 
				np.round(self.prediction, 3))), self.y))
			csvwriter.writerows(Agtron_result)
			
			info = np.array(['Calibration R square', 'RMSECV', 'Prediction R square', 
				'RMSEP', 'SEP', 'STD', 'RPD']).reshape(7, -1)
			result = np.array([np.round(self.cal_r2, 3), np.round(self.rmsecv, 3), 
				np.round(self.pre_r2, 3), np.round(self.rmsep, 3), np.round(self.sep, 3),
				np.round(self.std, 3), np.round(self.rpd, 3)]).reshape(7, -1)
			result = np.hstack((info, result))
			csvwriter.writerows(result)

def main():
	label, nir, agtron, _, _ = read_data()
	x = nir
	y = agtron
	#------------------------------------------------------ 取到B50
	test_split = list(range(6, 1500, 7))
	# print(test_split)
	train_split = list(i for i in range(0, 1500) if i not in test_split)
	# print(train_split)
	train_x, train_y, train_name, test_x, test_y, test_name = \
	x[train_split, :], np.expand_dims(y[train_split, 1], axis = 1), np.expand_dims(label[train_split, 2], axis = 1), \
	x[test_split, :], np.expand_dims(y[test_split, 1], axis = 1), np.expand_dims(label[test_split, 2], axis = 1)

	print(train_x.shape, train_y.shape, train_name.shape, \
		test_x.shape, test_y.shape, test_name.shape)

	all_preprocess = [msc, msc_sg1222, sg1222_msc]
	total_pc = 10

	for preprocess in all_preprocess:
		for wave in range(9):
			new_train_x, new_test_x = \
			preprocess(train_x)[:, wave_select(wave)], preprocess(test_x)[:, wave_select(wave)]
			trv = np.zeros(shape = (total_pc + 1, ))
			trv[0] = np.sum((train_y - np.mean(train_y, axis = 0)) ** 2, axis = 0) / (train_y.shape[0])	
			for cpt in range(1, total_pc + 1):
				print(f'Preprocess: {preprocess.__name__}\tWave: {wave}\tComponent: {cpt}')
				plsr = Plsr(cpt)
				trv[cpt] = plsr.train(new_train_x, train_y)
				plsr.predict(new_test_x, test_y)
				plsr.save_to_csv(test_name, preprocess.__name__, wave)

				if cpt == total_pc:
					plsr.optimal_pc(trv)


if __name__ == '__main__':
	main()

