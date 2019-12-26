import csv
import numpy as np
import pandas as pd

from itertools import combinations

if __name__ == '__main__':
	flavor_name = ['Floral', 'Fruity', 'Sour/Fermented', 'Vegetable', 'Other', 'Roasted', 'Spices', 'Nutty/Cocca', 'Sweet']
	data_process = np.load('./outputdata/process_a1_b1_b33.npy', allow_pickle = True)
	data = np.load('./outputdata/flavor_a1_b1_b33.npy', allow_pickle = True)
	big_data = np.hstack((data, data_process))
	big_data = big_data[~pd.isnull(big_data).any(axis = 1)]

	flavor = list(range(0, 9))
	flavor_dict = {k:v for k, v in enumerate(flavor)}
	# test = [0, 1, 1, 0, 0, 0, 0, 0, 0]
	# g = [k for k, v in flavor_dict.items() if test[k] == 1]
	# print(tuple(g))
	# print(len(big_data[:, 9]))
	data_n = np.vstack((big_data[:, 2], big_data[:, 8])).T
	data_v = np.hstack((data_n, big_data[:, 11:18])).astype(np.int)
	n_data = []
	for i in data_v:
		n = [k for k, v in flavor_dict.items() if i[k] == 1]
		n_data.append(tuple(n))

	for i in range(1, 10, 1):
		count = 0
		flav_per = list(combinations(flavor_name, i))
		per = list(combinations(flavor, i))
		with open(f'./outputdata/flavorClass/num_of_{i}_flavor.csv', 'w', newline = '') as f:
			w = csv.writer(f)
			w.writerow(['Samplename', 'Number', 'Process', 'Combinations'])
			for j in range(len(per)):
				for k in range(len(n_data)):
					if per[j] == n_data[k]:
						count += 1
						w.writerow([big_data[k][0], big_data[k][1], int(big_data[k][-1]), flav_per[j]])
				w.writerow([])
			w.writerow(['Total num:', count])