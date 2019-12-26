import numpy as np
import pandas as pd
import sys
import argparse
import glob

def dataCollect(args):
	
	df = pd.read_excel(args.path)

	#Change type of columns' names into string, eg. 700 -> '700'
	df.columns = df.columns.astype(str)

	#Whether the user wants average data or not
	if args.a == True:
		undo = df.loc[df['Batch'] <= 9, :]
		lower = df.loc[df['Batch'] > 9, :]
		lower_rps = lower.loc[range(len(undo), len(df), 3), 'Sample_name':'Sweet']
		lower_nir = lower.loc[:, '700':'2498']
		lower_rps = lower_rps.reset_index()
		l = []
		for i in range(len(undo), len(df), 3):
			l.append(lower_nir.loc[i:i + 2, :].mean(axis = 0))
		lower_avg_nir = pd.concat(l, sort = False, axis = 1).T
		lower_new = pd.concat([lower_rps, lower_avg_nir], sort = False, axis = 1)
		lower_new = lower_new.iloc[:, 1:]
		df = pd.concat([undo, lower_new], sort = False, axis = 0)

	#Concatenate the name and number into data
	name_number = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), ['Sample_name', 'Number']]

	if args.target in ['Agtron', 'agtron']:
		data = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), 'Agtron']
	if args.target in ['Country', 'country']:
		data = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), 'Country']
	if args.target in ['Process', 'process']:
		data = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), 'Process']
	if args.target in ['Flavor', 'flavor']:
		data = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), 'Floral':'Sweet']

	if args.target in ['Nir', 'nir']:
		data = df.loc[(df['Batch'] >= args.bs) & (df['Batch'] <= args.be), f'{args.ws}':f'{args.we}']

	data = pd.concat([name_number, data], sort = False, axis = 1)
	print(data)
	data = data.values

	#Save data as .npy(When load it, 'allow_pickle' in np.load should be True)
	#Eg. a = np.load('../outputdata/nir_a0_b5_b20_1500_2000.npy', allow_pickle = True)
	if args.ws and args.we and args.target in ['NIR', 'nir', 'FTIR', 'ftir']:
		np.save(f'../../outputdata/{args.target}_a{args.a}_b{args.bs}_b{args.be}_{args.ws}_{args.we}.npy', data)
	else:
		np.save(f'../../outputdata/{args.target}_a{args.a}_b{args.bs}_b{args.be}.npy', data)
	print('Data are saved as npy !!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'DataCollector Tutorial')
	parser.add_argument('-p', '--path', type = str, help = '[Input] Data location, default is set.', default = glob.glob("../../data/*.xlsx")[0])
	parser.add_argument('-t', '--target', type = str, required = True, help = '[Output] Data you want, eg. Agtron, Country, Process, Flavor, Nir or Ftir spectrum.')
	parser.add_argument('-a', '--average', type = int, default = 0, help = 'Average your data or not. Default is false', dest = 'a')
	parser.add_argument('-bs', '--batch_start', type = int, required = True, dest = 'bs')
	parser.add_argument('-be', '--batch_end', type = int, required = True, dest = 'be')
	parser.add_argument('-ws', '--wave_start', type = int or float, dest = 'ws', default = 700)
	parser.add_argument('-we', '--wave_end', type = int or float, dest = 'we', default = 2498)
	args = parser.parse_args()
	dataCollect(args)

	