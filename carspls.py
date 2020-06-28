import os
import sys
import csv
import math
import ipdb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.utils.package import read_data
from lib.utils.wave_preprocess import raw, msc, sg1222_msc, msc_sg1222, wave_select

from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

def random_sample(ratio, sample_size):
	choose = list(np.random.randint(sample_size, size = int(ratio * sample_size)))
	return choose

class Carspls():
	def __init__(self, n_components, num_run):
		
		self.n_components = n_components
		self.plsr = PLSRegression(n_components = self.n_components)
		self.num_run = num_run #num of sampling runs

	def train(self, x, y, sample, wave, run, replace):
		self.p = 900
		# print('Old wave number', len(wave))
		if run == 1:
			self.w = None
		wave = sorted(list(dict.fromkeys(self.adaptively_reweighted_sampling(wave, int(self.retained_ratio(run) * self.p), self.w, replace))))
		# print('New wave number', len(wave))
		x = x[sample, :]
		x = x[:, wave]
		y = y[sample]
		if len(wave) < self.n_components:
			print('End of fitting')
			sys.exit(1)
		self.plsr.fit(x, y)
		b = abs(self.plsr.coef_)
		w = b / np.sum(b)
		self.w = w.squeeze().tolist()
		self.rmsecv = 0.0
		self.cal_r2 = 0.0
		all_pred_y = np.zeros(shape = (len(sample), 1))
		#Leave-one-out
		for i in range(len(sample)):
			#Split calibration set and validation set
			calibration_idx = [idx for idx in range(len(sample)) if idx != i]
			valid_x = x[i, :].reshape(1, -1)
			valid_y = y[i, :]
			calibration_x = x[calibration_idx, :]
			calibration_y = y[calibration_idx, :]

			#Validate the data
			pred_y = self.plsr.predict(valid_x)
			all_pred_y[i] = pred_y
			self.rmsecv += (valid_y - pred_y) ** 2

		self.rmsecv /= len(sample)
		self.rmsecv **= 0.5
		self.cal_r2 = r2_score(y, all_pred_y)
		return wave

	def predict(self, x, y, wave):
		self.y = y
		m = self.y.shape[0]

		self.prediction = self.plsr.predict(x[:, wave])

		bias = np.sum(np.subtract(self.prediction, self.y), axis = 0) / m
		self.sep = ((np.sum(np.subtract(np.subtract(self.prediction, self.y), bias) ** 2, axis = 0))/ 
				(m - 1)) ** 0.5
		self.rmsep = (np.sum(np.subtract(self.prediction, self.y) ** 2, axis = 0) / m) ** 0.5
		self.std = np.std(self.y, axis = 0)
		self.rpd = self.std / self.sep
		self.pre_r2 = r2_score(y, self.prediction)
		print(self.rpd)
		return self.rpd
		# print(self.sep, self.rmsep, self.std, self.rpd, self.pre_r2)

	def retained_ratio(self, i):
		if i == 1:
			return 1
		elif i == self.num_run:
			return 2/self.p
		else:
			return (self.p/2) ** (1 / (self.num_run - 1)) * math.exp(-math.log(self.p / 2) / (self.num_run - 1) * i)

	def adaptively_reweighted_sampling(self, wave, num_retained, propability, replace):
		return np.random.choice(wave, size = num_retained, p = propability, replace = replace)

	def optimal_pc(self, all_trv):
		formula = [all_trv[0] * 0.01 * x + all_trv[x] for x in range(1, all_trv.shape[0])]
		optimal_pc = formula.index(min(formula)) + 1
		print('Optimal :', optimal_pc)

	#Used to compute the optimal # of principal component
	def total_residual_variance(self, matrix):
		return np.sum((matrix - np.mean(matrix, axis = 0)) ** 2, axis = 0) / (matrix.shape[0])

	def save_to_csv(self, name, preprocessing, wavelength, folder_name, run, wave):
		path = f'./outputdata/{folder_name}/'
		if not os.path.exists(path):
			os.makedirs(path)
		with open(f'{path}/plsr_pc{self.n_components}_{preprocessing}_{wavelength}_run_{run}.csv', 'w', newline = '') as f:
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
			wave = [i*2 + 700 for i in wave]
			csvwriter.writerow(wave)

def main(args):
	label, nir, agtron, _, _ = read_data()
	s1 = '1028 1038 1058 1060 1064 1080 1082 1084 1086 1088 1094 1096 1098 1100 1102 1104 1106 1108 1110 1112 1114 1118 1120 1122 1124 1126 1130 1132 1134 1136 1138 1142 1144 1146 1148 1174 1176 1178 1180 1182 1184 1192 1244 1246 1292 1294 1298 1302 1304 1306 1308 1310 1322 1328 1364 1366 1404 1412 1420 1422 1548 1556 1558 1560 1562 1582 1590 1592 1594 1598 1600 1606 1616 1618 1620 1622 1626 1628 1630 1632 1634 1636 1638 1640 1642 1644 1646 1648 1650 1652 1654 1656 1658 1660 1666 1670 1674 1676 1682 1684 1686 1712 1722 1724 1726 1728 1730 1732 1734 1736 1740 1744 1746 1748 1750 1758 1760 1762 1768 1770 1772 1774 1778 1780 1788 1804 1818 2042 2046 2048 2050 2056 2058 2060 2062 2164 2168 2170 2218 2220 2224 2226 2228 2272 2298 2300 2302 2306 2308 2312 2332 2336 2342 2354 2356 2404 2496'.split(' ')
	s2 = '876 878 1070 1072 1074 1078 1104 1108 1110 1112 1114 1120 1122 1124 1126 1130 1132 1136 1138 1140 1146 1148 1150 1152 1154 1156 1158 1162 1164 1172 1176 1208 1210 1284 1286 1288 1290 1296 1332 1346 1350 1354 1438 1440 1444 1446 1448 1454 1476 1482 1504 1536 1664 1668 1674 1678 1680 1682 1688 1696 1698 1702 1720 1722 1724 1726 1730 1754 1760 1762 1764 1804 1806 1810 1812 1814 2036 2038 2042 2060 2062 2064 2066 2068 2132 2212 2214 2302 2332 2334 2374 2376 2382 2484 2490'.split(' ')
	s3 = '1090 1104 1108 1110 1120 1122 1126 1130 1142 1152 1158 1160 1168 1170 1282 1284 1290 1294 1298 1344 1356 1444 1632 1658 1670 1678 1680 1690 1692 1694 1702 1708 1712 1720 1762 1766 1810 1814 1838 1942 2060 2064 2066 2210 2294 2296 2302 2304 2308 2332 2370'.split(' ')
	x = nir
	y = agtron
	plt.plot(list(range(700, 2500, 2)), x[100])
	for xc in s1:
		plt.axvline(int(xc))
	plt.savefig('./s1.png')

	plt.clf()
	plt.plot(list(range(700, 2500, 2)), x[100])
	for xc in s2:
		plt.axvline(int(xc))
	plt.savefig('./s2.png')

	plt.clf()
	plt.plot(list(range(700, 2500, 2)), x[100])
	for xc in s3:
		plt.axvline(int(xc))
	plt.savefig('./s3.png')
	#------------------------------------------------------ 取到B50
	# test_split = list(range(6, 1500, 7))
	# # print(test_split)
	# train_split = list(i for i in range(0, 1500) if i not in test_split)
	# # print(train_split)
	# train_x, train_y, train_name, test_x, test_y, test_name = \
	# x[train_split, :], np.expand_dims(y[train_split, 1], axis = 1), np.expand_dims(label[train_split, 2], axis = 1), \
	# x[test_split, :], np.expand_dims(y[test_split, 1], axis = 1), np.expand_dims(label[test_split, 2], axis = 1)

	# print(train_x.shape, train_y.shape, train_name.shape, \
	# 	test_x.shape, test_y.shape, test_name.shape)

	# all_preprocess = {0:raw, 1:msc, 2:msc_sg1222, 3:sg1222_msc}
	# total_pc = args.pc
	# num_runs = args.rn
	# ratio = args.ra
	# folder_name = args.fn
	# preprocess = all_preprocess[args.ps]
	# replace = False # true means replicated happens

	# # for preprocess in all_preprocess:
	# new_train_x, new_test_x = preprocess(train_x), preprocess(test_x)
	# # trv = np.zeros(shape = (total_pc + 1, ))
	# # trv[0] = np.sum((train_y - np.mean(train_y, axis = 0)) ** 2, axis = 0) / (train_y.shape[0])	
	# # for cpt in range(1, total_pc + 1):
	# print(f'Preprocess: {preprocess.__name__}\tComponent: {total_pc}')
	# plsr = Carspls(total_pc, num_runs)
	# rpd = np.zeros(shape = (num_runs, ))
	# rpd_best = 0.0
	# choose_wave = list(range(new_train_x.shape[1]))
	# for run in range(1, num_runs):
	# 	# print(f'{run}/{num_runs}\t')
	# 	choose_sample = random_sample(ratio, new_train_x.shape[0])
	# 	choose_wave = plsr.train(new_train_x, train_y, choose_sample, choose_wave, run, replace)
	# 	rpd[run] = plsr.predict(new_test_x, test_y, choose_wave)
	# 	plsr.save_to_csv(test_name, preprocess.__name__, len(choose_wave), folder_name, run, choose_wave)
	# 	plt.plot(np.arange(num_runs), rpd)
	# 	plt.savefig(f'./outputdata/{folder_name}/cars_{preprocess.__name__}_pc{total_pc}_runs{num_runs}.png')
	# 	if rpd[run] > rpd_best:
	# 		rpd_best = rpd[run]
	# 		print(f"at run: {run} the best rpd is {rpd_best}")
	# 	# if cpt == total_pc:
	# 	# 	plsr.optimal_pc(trv)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'CARSPLSR')
	parser.add_argument('-pc', '-principal_components', type = int)
	parser.add_argument('-rn', '-runs', type = int)
	parser.add_argument('-ra', '-ratio', type = float)
	parser.add_argument('-ps', '-preprocessing', type = int, choices = [0, 1, 2, 3], help = "raw for 0, msc for 1, msc sg for 2, sg msc for 3")
	parser.add_argument('-fn', '-folder_name', type = str)
	
	args = parser.parse_args()
	print(args)
	
	with ipdb.launch_ipdb_on_exception():
		sys.breakpointhook = ipdb.set_trace
		main(args)