import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.preprocessing.preprocess import MSC, SG
from sklearn.decomposition import PCA



df = pd.read_excel('./data/coffee_nir_new_flavor_20200315_ftir.xlsx', encoding = 'big5')
col = df.columns.values

ftir_wavelength = col[932:] # num # 2800 - 932
ftir_with_no_nan = df.dropna(subset = df.columns[932:]).values

agtron_value = ftir_with_no_nan[:, 3]
ftir_absorbance = ftir_with_no_nan[:, 932:].astype(np.float64)
msc_ftir, _ = MSC(ftir_absorbance)


start_time = time.time()

df = pd.read_excel('./data/coffee_NIR_new_flavor_20200321.xlsx', encoding = 'big5')


# pca = PCA(10)
# X_r = pca.fit(msc_ftir).transform(msc_ftir)
# print('explained variance ratio : %s'
# 	% str(pca.explained_variance_ratio_))


# plt.figure()
# plt.xlabel(f'PC1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)')
# plt.ylabel(f'PC3 ({round(pca.explained_variance_ratio_[2] * 100, 2)}%)')
# # for i in range(X_r.shape[0]):
# plt.scatter(X_r[:, 0], X_r[:, 2], c = agtron_value, cmap = 'rainbow', alpha = .8, lw = 2)
# plt.colorbar()
# # plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
# plt.title(f'PCA')
# plt.tight_layout()
# plt.show()
# print(ftir_wavelength)

# data = df.values
# print(col)
# print(data[:5])