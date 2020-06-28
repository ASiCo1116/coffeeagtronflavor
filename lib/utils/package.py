import os
import time
import numpy as np
import pandas as pd

from tqdm import tqdm

def read_data(input_path = 'lib/data/coffee_NIR_new_flavor_20200507.xlsx'):
	print(f'Loading data...')
	start_time = time.time()
	df = pd.read_excel(input_path, encoding = 'utf-8')

	label = df.iloc[:, [0, 1, 4]]
	agtron = df.iloc[:, [2, 3]]
	flavor = df.iloc[:, 10:31]
	nir = df.iloc[:, 32:]

	not_nan_loc = np.where(~np.isnan(df.iloc[:, 10]))[0]
	flavor = flavor.iloc[not_nan_loc, :]
	f9 = _flavor21_to_flavor9(flavor)

	print(f'label size: {label.shape}')
	print(f'nir size: {nir.shape}')
	print(f'agtron size: {agtron.shape}')
	print(f'flavor9 size: {f9.shape}')
	print(f'flavor21 size: {flavor.shape}')
	print(f'Loading data costs {time.time() - start_time} s')
	print('Finish loading')

	return label.to_numpy(), nir.to_numpy(), agtron.to_numpy(), f9.to_numpy(), flavor.to_numpy().astype(int)

def _flavor21_to_flavor9(flavor):
	f9 = pd.DataFrame({
		'Floral' : [], 'Fruity': [], 'S/F' : [], 'G/V' : [], 'Other' : [],
		'Roasted' : [], 'Spices' : [], 'N/C': [], 'Sweet' : []})

	f9['Floral'] = np.where(np.sum(flavor.iloc[:, 0:2], axis = 1) > 0, 1, 0)
	f9['Fruity'] = np.where(np.sum(flavor.iloc[:, 2:7], axis = 1) > 0, 1, 0)
	f9['S/F'] = np.where(np.sum(flavor.iloc[:, 7:10], axis = 1) > 0, 1, 0)
	f9['G/V'] = np.where(np.sum(flavor.iloc[:, 10:12], axis = 1) > 0, 1, 0)
	f9['Other'] = np.where(np.sum(flavor.iloc[:, 12:14], axis = 1) > 0, 1, 0)
	f9['Roasted'] = np.where(np.sum(flavor.iloc[:, 14:16], axis = 1) > 0, 1, 0)
	f9['Spices'] = flavor.iloc[:, 16].reset_index(drop = True).astype(int)
	f9['N/C'] = np.where(np.sum(flavor.iloc[:, 17:19], axis = 1) > 0, 1, 0)
	f9['Sweet'] = np.where(np.sum(flavor.iloc[:, 19:21], axis = 1) > 0, 1, 0)

	return f9


