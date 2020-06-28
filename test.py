import numpy as np

from lib.utils.package import read_data
from lib.utils.wave_preprocess import raw, msc, sg1222_msc, msc_sg1222, wave_select
from lib.utils.metrics import f1score_rec_acc, print_number_ratio, sample_to_spectra

from sklearn.model_selection import train_test_split
from functools import reduce
from IPython import embed

def main():

	label, nir, agtron, f9, f21 = read_data()
	x, label, y1, y2 = nir[302:-111], label[302:-111], f9[178:-111], f21[178:-111]

	sf = np.where(y1[:, 2] == 1)[0]
	spices = np.where(y1[:, 6] == 1)[0]
	other = np.where(y1[:, 4] == 1)[0]
	fruity = np.where(y1[:, 1] == 0)[0]
	sweet = np.where(y1[:, 8] == 0)[0]

	ssfs = reduce(np.union1d, (sf, spices, fruity, sweet))
	ssfs = [i//3 for i in ssfs if i % 3 == 0]

	embed()

if __name__ == '__main__':
	main()