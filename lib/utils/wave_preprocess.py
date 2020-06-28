import sys
from .preprocess import MSC, SG

def raw(x):
	return x

def msc(x):
	x, _ = MSC(x)
	return x

def sg1222_msc(x):
	x = SG(x, 'SG_w5_p2_d1')
	x, _ = MSC(x)
	return x

def msc_sg1222(x):
	x, _ = MSC(x)
	x = SG(x, 'SG_w5_p2_d1')
	return x

def wave_select(model):
	n1 = [w for w in range(101)]

	if model == 0:
		return n1 
	elif model == 1:
		n1.extend([w for w in range(230, 281)])
		return n1
	elif model == 2:
		n1.extend([w for w in range(350, 401)])
		return n1
	elif model == 3:
		n1.extend([w for w in range(500, 551)])
		return n1
	elif model == 4:
		n1.extend([w for w in range(600, 651)])
		return n1
	elif model == 5:
		n1.extend([w for w in range(725, 776)])
		return n1
	elif model == 6:
		n1.extend([w for w in range(825, 876)])
		return n1
	elif model == 7:
		return [w for w in range(0, 900)]
	elif model == 8:
		return [w for w in range(200, 900)]
	else:
		print("ERROR!")
		sys.exit(1)