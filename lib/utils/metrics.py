import numpy as np
import matplotlib.pyplot as plt

########################################

# only suitable for triplicate spectra

########################################

col21_name = 'Floral, Tea-like, Tropical Fruit, Stone Fruit, Citrus Fruit, Berry Fruit, Other Fruit, Sour, Alcohol, Fermented, Fresh Vegetable, Dry Vegetable, Papery/Musty, Chemical, Burnt, Cereal, Spices, Nutty, Cocca, Sweet, Butter/Milky'.split(', ')
col9_name = 'Floral, Fruity, S/F, G/V, Other, Roasted, Spices, N/C, Sweet'.split(', ')

def f1score_rec_acc(y_pred, y_true, verbose):

	total_acc = round(np.sum(y_pred == y_true) / (y_pred.shape[0] * y_pred.shape[1]), 3)
	class_acc = np.zeros(shape = (y_pred.shape[1], ))
	
	total_rec = round(np.sum(y_pred & y_true) / np.sum(y_true), 3)
	class_rec = np.zeros(shape = (y_pred.shape[1], ))
	
	total_pre = round(np.sum(y_pred & y_true) / np.sum(y_pred), 3)
	class_pre = np.zeros(shape = (y_pred.shape[1], ))

	f1_score = round(2 * total_pre * total_rec / (total_pre + total_rec), 3)

	for i in range(y_pred.shape[1]):
		class_acc[i] = np.around(np.sum(y_pred[:, i] == y_true[:, i]) / y_pred.shape[0], 3)
		
		if int(np.sum(y_true[:, i])) != 0:
			class_rec[i] = np.around(np.sum(y_pred[:, i] & y_true[:, i]) / np.sum(y_true[:, i]), 3)
		else:
			class_rec[i] = .0

		if int(np.sum(y_pred[:, i])) != 0:
			class_pre[i] = np.around(np.sum(y_pred[:, i] & y_true[:, i]) / np.sum(y_pred[:, i]), 3)
		else:
			class_pre[i] = .0

	print(f'F1 score: {f1_score}'.rjust(25), end = '')
	print(f'Total Rec: {total_rec}'.rjust(25), end = '')
	print(f'Total Acc: {total_acc}'.rjust(25))

	if verbose == 1:

		if y_pred.shape[1] == 9:
			print('    ', end = '')
			for i in range(y_pred.shape[1]):
				print(col9_name[i].rjust(15), end = '')
			print('\nRec:', end = '')
			for i in range(y_pred.shape[1]):
				print(str(class_rec[i]).rjust(15), end = '')
			print('\nAcc:', end = '')
			for i in range(y_pred.shape[1]):
				print(str(class_acc[i]).rjust(15), end = '')

		if y_pred.shape[1] == 21:
			for i in range(3):
				print('    ', end = '')
				for j in range(7):
					print(col21_name[i * 7 + j].rjust(18), end = '')
				print('\nRec:', end = '')
				for j in range(7):
					print(str(class_rec[i * 7 + j]).rjust(18), end = '')
				print('\nAcc:', end = '')
				for j in range(7):
					print(str(class_acc[i * 7 + j]).rjust(18), end = '')
				print('\n')	

	return f1_score, total_rec, total_acc

def print_number_ratio(y_part, y_total):

	part_number = np.sum(y_part, axis = 0).astype(int)
	total_number = np.sum(y_total, axis = 0).astype(int)
	ratio = np.around(part_number / total_number, 3)

	if y_part.shape[1] == 9:
		print('      ', end = '')
		for i in range(y_part.shape[1]):
			print(col9_name[i].rjust(15), end = '')
		print('\nPart: ', end = '')
		for i in part_number:
			print(str(i).rjust(15), end = '')
		print('\nTotal:', end = '')
		for i in total_number:
			print(str(i).rjust(15), end = '')
		print('\nRatio:', end = '')
		for i in ratio:
			print(str(i).rjust(15), end = '')
		print('\n')

	if y_part.shape[1] == 21:
		for i in range(3):
			print('      ', end = '')
			for j in range(7):
				print(col21_name[i * 7 + j].rjust(18), end = '')
			print('\nPart: ', end = '')
			for j in range(7):
				print(str(part_number[i * 7 + j]).rjust(18), end = '')
			print('\nTotal:', end = '')
			for j in range(7):
				print(str(total_number[i * 7 + j]).rjust(18), end = '')
			print('\nRatio:', end = '')
			for j in range(7):
				print(str(ratio[i * 7 + j]).rjust(18), end = '')
			print('\n')

def sample_to_spectra(index):
	index_ = [i * 3 + j for j in range(3) for i in index]
	return index_


def plot_f1_rec_acc(x_parameters, y_metrics, metrics_name, xlabel, fig_name, title_name):
	for m in range(len(y_metrics)):
		plt.plot(x_parameters, y_metrics[m], label = metrics_name[m])

	plt.legend()
	plt.ylabel('F1 score')
	plt.xlabel(f'{xlabel}')
	plt.title(f'{title_name}')
	plt.savefig(f'./{fig_name}.png')
	plt.clf()


def save_model(model):
	return None




