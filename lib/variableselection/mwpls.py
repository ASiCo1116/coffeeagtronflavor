import sys
import numpy as np
import matplotlib.pyplot as plt

from model.plsr import Plsr

class Mwpls(object):
	def __init__(self, name_matrix, train_x, train_y, test_x, test_y, num_principal_components, preprocessing, window_size = 11):
		#Num of variables/features/wavenumbers
		self.ws = window_size if window_size % 2 == 1 else sys.exit('Window size need to be odd number!')
		
	def main(self):
		#Half window size
		w = (self.ws - 1) / 2
		self.results = np.zeros((8, 900))
		for midpoint in range(w, 900 - w): #900 wavenumbers
			window = np.aragne(midpoint - w, midpoint + w)
			plsr = Plsr(name_matrix, train_x[:, window], train_y, test_x[:, window], test_y, num_principal_components)
			self.results[0][midpoint], self.results[1][midpoint], self.results[2][midpoint] = plsr.train()
			self.results[3][midpoint], self.results[4][midpoint], self.results[5][midpoint], self.results[6][midpoint], self.results[7][midpoint] = plsr.predict()

		return self.results

	def plots(self):
		x = np.arange(700, 2498, 2)

		plt.title(f'123')
		plt.plot(x, self.results[2])
		plt.xlabel('Wavelength (nm)')
		plt.ylabel(f'RMSECV')
		plt.savefig(f'/Users/mengchienhsueh/Desktop/{123}.png')
		plt.show()