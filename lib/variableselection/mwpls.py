import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/mengchienhsueh/EVERYTHING/CoffeeProgram/lib/')
from model.plsr import Plsr

class Mwpls(object):
	def __init__(self, name_matrix, train_x, train_y, test_x, test_y, num_principal_components, preprocessing, wavelength, window_size = 11):
		#Num of variables/features/wavenumbers
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.name_matrix = name_matrix
		self.n_pcts = num_principal_components
		self.preprocessing = preprocessing
		self.wavelength = wavelength
		self.ws = window_size if window_size % 2 == 1 else sys.exit('Window size need to be odd number!')
		
	def main(self):
		#Half window size
		w = int((self.ws - 1) / 2)
		self.results = np.zeros((8, 900))
		for midpoint in range(w, 900 - w): #900 wavenumbers
			window = np.arange(midpoint - w, midpoint + w)
			plsr = Plsr(self.name_matrix, self.train_x[:, window], self.train_y, self.test_x[:, window], self.test_y, self.n_pcts, self.preprocessing, self.wavelength)
			self.results[0][midpoint], self.results[1][midpoint], self.results[2][midpoint] = plsr.train()
			self.results[3][midpoint], self.results[4][midpoint], self.results[5][midpoint], self.results[6][midpoint], self.results[7][midpoint] = plsr.predict()

		return self.results

	def plots(self):
		x = np.arange(700, 2498, 2)

		plt.title(f'123')
		plt.plot(x, self.results[2])
		plt.xlabel('Wavelength (nm)')
		plt.ylabel(f'RMSECV')
		# plt.savefig(f'/Users/mengchienhsueh/Desktop/{123}.png')
		plt.show()

if __name__ == '__main__':
	x = np.random.random((10, 900))
	y = np.random.random((10, 1))
	name = 'abc'
	mwpls = Mwpls(name, x, y, x, y, 7, 'as', 'ab')
	mwpls.main()
	mwpls.plots()

