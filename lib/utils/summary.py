import numpy as np

def plsr_summary(train, predict):
	optimal_pc, c_r2, rmsecv = train
	p_r2, rmsep, sep, p_std, p_rpd = predict
	fields = [('Principal Component', *list(range(1, len(c_r2) + 1))),
			('Optimal principal component', *list('V' for i in range(1, len(c_r2) + 1) if i == optimal_pc)),
			('Root mean square error of CV', *rmsecv)
			]
	lengths = [30]
	fmt = ' '.join('{:<%d}' % l for l in lengths)
	print(fmt.format(*fields[0]))
	print('-' * (sum(lengths) + len(lengths) - 1))  # separator
	for row in fields[1:]:
		print(fmt.format(*row))
	# print('=' * 80)
	# print(f'Optimal principal component: {optimal_pc}')
	# print('-' * 80)
	# print('                ', *list(i for i in range(1, len(c_r2) + 1)), sep = '     ')
	# print('Cal- R square:   {:.3f}').format(*c_r2) for in enumerate(c_r2)