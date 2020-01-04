import numpy as np

def train_test_split(x, y, start, ratio):
	step = int(1 / ratio) - 1
	test_idx = list(range(start, x.shape[0], step))
	train_x = x[[i for i in range(x.shape[0]) if i not in test_idx], :]
	train_y = y[[i for i in range(x.shape[0]) if i not in test_idx], :]
	test_x = x[test_idx, :]
	test_y = y[test_idx, :]

	return train_x, train_y, test_x, test_y