import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNearestNeighobrs():
	def __init__(self, model_name):
		self.modelName = model_name
		if type(model_name) == str:
			if model_name.endswith('.pkl'):
				f = open(self.modelName, 'rb')
				self.model = pickle.load(f)


	def train(self, data, label):



	def predict(self, predict):

	


