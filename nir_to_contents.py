import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from lib.model.plsr import Plsr
from lib.preprocessing.preprocess import MSC, SG
from utils.summary import plsr_summary

def wavenumber_to_index(wave):
	return int((wave - 700) / 2)

def read_scores(label):
	df = pd.read_excel('./data/Chemicals_scores.xlsx')
	df = df.groupby(df['Number'], sort = False).mean()
	scores = df.iloc[:, label].values
	return scores

def read_ftir():
	df = pd.read_excel('./data/Chemical_ftir.xlsx')
	df = df.values
	wave = df[:, 1:]
	wave = np.vsplit(wave, wave.shape[0] / 3)
	wave = np.vstack(np.mean(wave, axis = 1))
	return wave.astype(np.float32)

def read_nir():  #[0:36]single [36:]triplicate
	df = pd.read_excel('./data/Chemical_nir.xlsx')
	df = df.values
	agtron1 = df[:36, 1]
	idx = [i for i in range(36, df.shape[0], 3)]
	agtron2 = df[idx, 1]
	agtron = np.hstack((agtron1, agtron2))
	wave1 = df[:36, 3:]
	wave2 = df[36:, 3:]
	wave2 = np.vsplit(wave2, wave2.shape[0] / 3)
	wave2 = np.vstack(np.mean(wave2, axis = 1))
	wave = np.vstack((wave1, wave2))
	return agtron.astype(np.float32), wave.astype(np.float32)

def read_contents():
	df = pd.read_excel('./data/Chemical_contents.xlsx')
	df = df.values
	name = df[:-1, 1]
	contents = df[:-1, 2:11]
	contents[contents == '-'] = 0.0
	return name.reshape(name.shape[0], 1), contents.astype(np.float32)

def pca_plot(pcs, X_matrix, Y_matrix, Name_matrix = None):
	pca = PCA(pcs)
	X_r = pca.fit(X_matrix).transform(X_matrix)
	Y_matrix = Y_matrix.reshape(Y_matrix.shape[0], )
	fig, ax = plt.subplots()
	print(f'explained variance ratio (first two components): {str(pca.explained_variance_ratio_)}')
	ax0 = ax.scatter(X_r[:, 0], X_r[:, 1], c = Y_matrix, alpha = .8, lw = 2, cmap = 'rainbow')
	ax.set_xlabel(f'PC1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)')
	ax.set_ylabel(f'PC2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)')
	# plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
	if Name_matrix != None:
		for i, txt in enumerate(Name_matrix.squeeze()):
			ax.annotate(txt, (X_r[i, 0], X_r[i, 1]))
	ax.set_title(f'PCA')
	fig.colorbar(ax0, ax = ax)
	# cbar.ax.set_xlabel('Agtron')
	# plt.tight_layout()
	plt.show()

def plot_wave(wave, type_ = 'nir'):
	if type_ == 'nir':
		for i in range(wave.shape[0]):
			plt.plot(np.arange(700, 2500, 2), wave[i])
	else:
		for i in range(wave.shape[0]):
			plt.plot(np.linspace(399.264912, 4000.364384, 1868), wave[i])
			if i == 0:
				break
	plt.show()

def wave_convert(wave, index):

	return wave

def standard(data):
	mean = np.mean(data, axis = 0)
	data -= mean
	return data

def plot_pca(pcs, data, label):
	pca = PCA(pcs)
	X_r = pca.fit(data).transform(data)
	print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))
	plt.figure()
	plt.xlabel(f'PC1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)')
	plt.ylabel(f'PC2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)')
	plt.scatter(X_r[:, 0], X_r[:, 1], c = label, cmap = 'rainbow', alpha = .8, lw = 2)
	plt.colorbar()
	plt.title(f'PCA')
	plt.tight_layout()
	plt.show()

	return None

def main():
	# ftir_wave = read_ftir()	
	agtron, nir_wave = read_nir()
	# agtron = agtron.reshape(agtron.shape[0], 1)
	name, contents = read_contents()
	nir_wave, _ = MSC(nir_wave)
	contents = standard(contents[:, 1])
	# print(contents)
	total_scores = read_scores(-1)
	# plot_pca(10, nir_wave, agtron)
	print(np.corrcoef(agtron, read_scores(3)))

	# chosen_wave = list(range(700, 900, 2)) + list(range(1600, 1750, 2)) #caffeine
	# chosen_wave = list(range(700, 900, 2)) + list(range(1100, 1200, 2)) + \
	# 	list(range(1300, 1330, 2)) + list(range(1400, 1450, 2)) +\
	# 	list(range(1600, 1750, 2)) + list(range(1900, 1950, 2)) +\
	# 	list(range(2200, 2350, 2)) + list(range(2400, 2450, 2)) #caffeine
	# chosen_wave = list(range(1400, 1600, 2)) + list(range(1900, 1980, 2)) + \
	# 	list(range(1400, 1450, 2)) + list(range(1600, 1750, 2)) + list(range(1900, 1950, 2)) +\
	# 	list(range(2200, 2350, 2)) + list(range(2400, 2450, 2)) #trigonelline
	# msc_ftir_wave, _ = MSC(ftir_wave)
	# plot_wave(msc_ftir_wave, 'ftir')
	# chosen_wave = list(map(wavenumber_to_index, chosen_wave))
	# nir_wave = nir_wave[:, chosen_wave]
	# msc_nir_wave, _ = MSC(nir_wave)
	# std_msc_nir_wave = standard(msc_nir_wave)
	# trigonelline = contents[:, 0].reshape(contents.shape[0], 1)
	# chlorogenic = contents[:, 1].reshape(contents.shape[0], 1)
	# caffeine = contents[:, 2].reshape(contents.shape[0], 1)
	# quinic = contents[:, 3].reshape(contents.shape[0], 1)
	# pca_plot(2, std_msc_nir_wave, chlorogenic, Name_matrix = None)
	
	# plsr = Plsr(name, msc_ftir_wave, agtron, msc_ftir_wave, agtron, 10, None, None)
	# print(plsr.train())
	# print(plsr.predict())

if __name__ == '__main__':
	main()