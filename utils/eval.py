import csv
import numpy as np

def evaluate(true_y, pred_y):
	if true_y.shape != pred_y.shape:
		print('Shape Error!')
		return None

	recall_total = np.sum(true_y)
	acc_total = np.sum(np.ones_like(true_y))
	class_recall_total = np.sum(true_y, axis = 0)
	class_acc_total = np.sum(np.ones_like(true_y), axis = 0)
	recall_error = np.sum(np.maximum((true_y - pred_y), 0))
	acc_error = np.sum(np.abs(true_y - pred_y))
	class_recall_error = np.sum(np.maximum((true_y - pred_y), 0), axis = 0)
	class_acc_error = np.sum(np.abs(true_y - pred_y), axis = 0)
	recall = (1 - recall_error / recall_total) * 100
	acc = (1 - acc_error / acc_total) * 100
	class_recall = (1 - class_recall_error / class_recall_total) * 100
	class_acc = (1 - class_acc_error / class_acc_total) * 100

	return recall, acc, class_recall.tolist(), class_acc.tolist()