import csv
import numpy as np

from sklearn.cross_decomposition import PLSRegression

def calculate_vips(model):

	t = model.x_scores_
	w = model.x_weights_
	q = model.y_loadings_
	p, h = w.shape
	vips = np.zeros((p,))
	s = np.diag(np.matmul(np.matmul(np.matmul(t.T,t),q.T), q)).reshape(h, -1)
	total_s = np.sum(s)

	for i in range(p):
		weight = np.array([(w[i,j] / np.linalg.norm(w[:,j])) ** 2 for j in range(h)])
		vips[i] = np.sqrt(p * (np.matmul(s.T, weight))/total_s)

	return vips