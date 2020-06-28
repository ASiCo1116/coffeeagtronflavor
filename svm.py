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
from sklearn.svm import SVC
from functools import reduce

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

	kernels = ['rbf', 'linear', 'sigmoid']
	Cs = [2 ** i for i in range(0, 21)]

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

	if args.num_of_flavor == 21:
		y1 = y2

	if args.subset == 1:
		print('small set')
		print_number_ratio(y1[total], y1)

	print('train')
	print_number_ratio(y1[train], y1[total])
	print('valid')
	print_number_ratio(y1[valid], y1[total])
	print('test')
	print_number_ratio(y1[test], y1[total])

	if args.run_model == 1:

		label_train, label_valid, label_test, x_train, x_valid, x_test, y_train, y_valid, y_test = label[train], label[valid], label[test], nir[train], nir[valid], nir[test], y1[train], y1[valid], y1[test]
	
		for kernel in kernels:

			overfit = []
			train_f1 = []
			valid_f1 = []
			l_c = []
			i = 0

			for c in Cs:
				model = []
				train_res = []
				valid_res = []
				predict_res = []
				for flavor in range(y_train.shape[1]):
					clf = SVC(C = float(c), kernel = kernel)
					clf.fit(x_train, y_train[:, flavor])
					model.append(clf)
					train_res.append(model[flavor].predict(x_train))
					valid_res.append(model[flavor].predict(x_valid))
					clf.fit(np.concatenate((x_train, x_valid), 0), np.concatenate((y_train[:, flavor], y_valid[:, flavor]), 0))
					predict_res.append(clf.predict(x_test))

				train_res = np.array(train_res).T.astype(int)
				valid_res = np.array(valid_res).T.astype(int)
				predict_res = np.array(predict_res).T.astype(int)

				print(f'Kernel: {kernel}\tC: {c}'.rjust(80))
				f1t, rect, acct = f1score_rec_acc(train_res, y_train, 0)
				f1v, recv, accv = f1score_rec_acc(valid_res, y_valid, 0)
				f1test, rectest, acctest = f1score_rec_acc(predict_res, y_test, 1)

				train_f1.append(f1t)
				valid_f1.append(f1v)
				l_c.append(i)
				i += 1

			if args.plot_f1_score:
				plot_f1_rec_acc(l_c, [train_f1, valid_f1], ['training', 'validation'], 'C value', f'svm_{kernel}_subset{args.subset}_flavor{args.num_of_flavor}', f'svm_{kernel}')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'svm')
	parser = add_arguments(parser)

	args = parser.parse_args()

	print(args)

	main(args)