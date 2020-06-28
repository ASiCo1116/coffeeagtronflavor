import os
import csv
import argparse
import numpy as np
import pandas as pd

from lib.utils.package import read_data
from lib.utils.wave_preprocess import raw, msc, sg1222_msc, msc_sg1222, wave_select
from lib.utils.metrics import f1score_rec_acc, print_number_ratio, sample_to_spectra, plot_f1_rec_acc
from argument import add_arguments

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from functools import reduce

class Knn:
	def __init__(self):
		pass

	def train(self, x, y, n_neighbors, weights):
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.model = []
		for i in range(y.shape[1]):
			self.model.append(KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights))
			self.model[i].fit(x, y[:,i])

	def predict(self, x):
		self.prediction = np.zeros((x.shape[0], len(self.model)))
		for i in range(len(self.model)):
			self.prediction[:, i] = self.model[i].predict(x)

		return self.prediction

	def save_to_csv(self, name, y):
		total_acc = np.sum(y == self.prediction) / (y.shape[0] * y.shape[1])
		total_prec_up = 0
		total_prec_down = 0
		total_rec_up = 0
		total_rec_down = 0
		cla_acc = np.zeros(shape = (y.shape[1], ))
		cla_rec = np.zeros(shape = (y.shape[1], ))
		for i in range(y.shape[1]):
			acc_up = 0
			rec_up = 0
			rec_down = 0
			for j in range(y.shape[0]):
				if self.prediction[j][i] == 1:
					total_prec_down += 1
					if y[j][i] == 1:
						total_prec_up += 1
				if y[j][i] == self.prediction[j][i]:
					acc_up += 1
				if y[j][i]  == 1:
					if self.prediction[j][i] == 1:
						rec_up += 1
						rec_down += 1
					rec_down += 1
			total_rec_up += rec_up
			total_rec_down += rec_down
			cla_acc[i] = round(acc_up / y.shape[0], 3)
			cla_rec[i] = round(rec_up / rec_down, 3)
		col21_name = 'Floral, Tea-like, Tropical Friut, Stone Friut, Citrus Fruit, Berry Fruit, Other Fruit, Sour, Alcohol, Fermented, Fresh Vegetable, Dry Vegetable, Papery/Musty, Chemical, Burnt, Cereal, Spices, Nutty, Cocca, Sweet, Butter/Milky'.split(', ')
		col9_name = 'Floral, Fruity, S/F, G/V, Other, Roasted, Spices, N/C, Sweet'.split(', ')
		# print('Flavor'.rjust(15), 'Acc.'.rjust(15), 'Rec'.rjust(15))
		# for i in range(y.shape[1]):
		# 	print(col21_name[i].rjust(15), str(round(cla_acc[i], 3)).rjust(15), str(round(cla_rec[i], 3)).rjust(15))
		total_prec = total_prec_up/total_prec_down
		total_rec = total_rec_up/total_rec_down
		f1_score = 2 * total_prec * total_rec / (total_prec + total_rec)
		print('Total acc:'.rjust(15), str(round(total_acc, 3)).rjust(15))
		print('Total rec:'.rjust(15), str(round(total_rec, 3)).rjust(15))
		print('Total f1:'.rjust(15), str(round(f1_score, 3)).rjust(15))
		# print(name)
		# print(y)
		# print(self.prediction)

		path = './outputdata/0518knn21_raw/'
		if not os.path.exists(path):
			os.makedirs(path)
		with open(f'{path}/knn_{self.n_neighbors}_{self.weights}.csv', 'w', newline = '') as f:
			csvwriter = csv.writer(f)

			title = ['Sample name']
			title.extend(col21_name)
			csvwriter.writerow(title)
			name = name[:, 2].squeeze()
			for i in range(y.shape[0]):
				csvwriter.writerow(np.hstack((name[i], self.prediction[i, :].astype(np.int))))
				csvwriter.writerow(np.hstack((np.array(['ground truth']), y[i, :].astype(np.int))))
			csvwriter.writerow(np.hstack((np.array(['Acc.']), cla_acc)))
			csvwriter.writerow(np.hstack((np.array(['Rec.']), cla_rec)))
			csvwriter.writerow(np.hstack((np.array(['Total acc.']), total_acc)))
			csvwriter.writerow(np.hstack((np.array(['Total rec.']), total_rec)))
			csvwriter.writerow(np.hstack((np.array(['F1']), f1_score)))
		return total_acc, total_rec, f1_score


def main(args):
	label, nir, agtron, f9, f21 = read_data()
	x, label, y1, y2 = nir[302:-111], label[302:-111], f9[178:-111], f21[178:-111]

	sf = np.where(y1[:, 2] == 1)[0]
	spices = np.where(y1[:, 6] == 1)[0]
	other = np.where(y1[:, 4] == 1)[0]
	fruity = np.where(y1[:, 1] == 0)[0]
	sweet = np.where(y1[:, 8] == 0)[0]

	ssfs = reduce(np.union1d, (sf, spices, fruity, sweet))
	ssfs = [i//3 for i in ssfs if i % 3 == 0]
	sso = reduce(np.union1d, (sf, spices, other))
	sso = [i//3 for i in sso if i % 3 == 0]
	so = np.union1d(sf, spices)
	so = [i//3 for i in so if i % 3 == 0]

	total = list(range(399))
	seed1 = args.seed1 #1
	seed2 = args.seed2 #42
	shuffle = True

	
	if args.subset == 1:
		seed0 = args.seed0
		_, samll = train_test_split(total, test_size = 150, random_state = seed0, shuffle = shuffle)
		total = samll

	if args.subset == 2:
		total = sso

	if args.subset == 3:
		total = so

	if args.subset == 4:
		total = ssfs

	train, test = train_test_split(total, test_size = .2, random_state = seed1, shuffle = shuffle)
	train, valid = train_test_split(train, test_size = .2, random_state = seed2, shuffle = shuffle)
	train = sample_to_spectra(train)
	valid = sample_to_spectra(valid)
	test = sample_to_spectra(test)
	total = sample_to_spectra(total)

	low_nei = 2
	high_nei = int(len(test) / 3 * 2)

	if args.num_of_flavor == 21:
		y1 = y2

	if args.subset == 1:
		print('small')
		print_number_ratio(y1[total], y1)

	print('train')
	print_number_ratio(y1[train], y1[total])
	print('valid')
	print_number_ratio(y1[valid], y1[total])
	print('test')
	print_number_ratio(y1[test], y1[total])

	if args.run_model == 1:

		label_train, label_valid, label_test, x_train, x_valid, x_test, y_train, y_valid, y_test = label[train], label[valid], label[test], nir[train], nir[valid], nir[test], y1[train], y1[valid], y1[test]
		
		for weight in ['uniform', 'distance']:

			overfit = []
			train_f1 = []
			valid_f1 = []
			l_nei = []
			
			for neighbor in range(low_nei, high_nei + 1):

				print(f'Neighbor: {neighbor}\tWeight: {weight}'.rjust(80))
				knn = Knn()
				knn.train(x_train, y_train, neighbor, weight)
				f1t, rect, acct = f1score_rec_acc(knn.predict(x_train).astype(int), y_train, 0)
				f1v, recv, accv = f1score_rec_acc(knn.predict(x_valid).astype(int), y_valid, 0)
				knn.train(np.concatenate((x_train, x_valid), 0), np.concatenate((y_train, y_valid), 0), neighbor, weight)
				f1test, rectest, acctest = f1score_rec_acc(knn.predict(x_test).astype(int), y_test, 1)

				train_f1.append(f1t)
				valid_f1.append(f1v)
				l_nei.append(neighbor)

			if args.plot_f1_score:
				plot_f1_rec_acc(l_nei, [train_f1, valid_f1], ['training', 'validation'], 'Neighbor', f'knn_{weight}_subset{args.subset}_flavor{args.num_of_flavor}', f'knn_{weight}')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'knn')
	parser = add_arguments(parser)
	args = parser.parse_args()

	print(args)

	main(args)

