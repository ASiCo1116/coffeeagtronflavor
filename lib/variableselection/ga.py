# import random

# import numpy as np
# from deap import base
# from deap import creator
# from deap import tools
# from sklearn import datasets
# from sklearn import model_selection
# from sklearn.cross_decomposition import PLSRegression

# # settings
# number_of_population = 100
# number_of_generation = 100
# max_number_of_components = 10
# fold_number = 5
# probability_of_crossover = 0.5
# probability_of_mutation = 0.2
# threshold_of_variable_selection = 0.5

# # generate sample dataset
# X_train, y_train = datasets.make_regression(n_samples=100, n_features=300, n_informative=10, noise=10, random_state=0)
# # print(type(X_train))

# X_train = np.load('./outputdata/nir_a0_b0_b39_700_2498.npy', allow_pickle = True)
# y_train = np.load('./outputdata/agtron_a0_b0_b39.npy', allow_pickle = True)
# X_train = X_train[:, 2:].astype(np.float64)
# y_train = y_train[:, 2:].astype(np.float64)

# # autoscaling
# autoscaled_X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0, ddof=1)
# autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# # GAPLS
# creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
# creator.create('Individual', list, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()
# min_boundary = np.zeros(X_train.shape[1])
# max_boundary = np.ones(X_train.shape[1]) * 1.0


# def create_ind_uniform(min_boundary, max_boundary):
#     index = []
#     for min, max in zip(min_boundary, max_boundary):
#         index.append(random.uniform(min, max))
#     return index


# toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
# toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
# toolbox.register('population', tools.initRepeat, list, toolbox.individual)


# def evalOneMax(individual):
#     individual_array = np.array(individual)
#     selected_X_variable_numbers = np.where(individual_array > threshold_of_variable_selection)[0]
#     selected_autoscaled_X_train = autoscaled_X_train[:, selected_X_variable_numbers]
#     if len(selected_X_variable_numbers):
#         # cross-validation
#         pls_components = np.arange(1, min(np.linalg.matrix_rank(selected_autoscaled_X_train) + 1,
#                                           max_number_of_components + 1), 1)
#         r2_cv_all = []
#         for pls_component in pls_components:
#             model_in_cv = PLSRegression(n_components=pls_component)
#             estimated_y_train_in_cv = np.ndarray.flatten(
#                 model_selection.cross_val_predict(model_in_cv, selected_autoscaled_X_train, autoscaled_y_train,
#                                                   cv=fold_number))
#             estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()
#             r2_cv_all.append(1 - sum((y_train - estimated_y_train_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))
#         value = np.max(r2_cv_all)
#     else:
#         value = -999

#     return value,


# toolbox.register('evaluate', evalOneMax)
# toolbox.register('mate', tools.cxTwoPoint)
# toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
# toolbox.register('select', tools.selTournament, tournsize=3)

# # random.seed(100)
# random.seed()
# pop = toolbox.population(n=number_of_population)

# print('Start of evolution')

# fitnesses = list(map(toolbox.evaluate, pop))
# for ind, fit in zip(pop, fitnesses):
#     ind.fitness.values = fit

# print('  Evaluated %i individuals' % len(pop))

# for generation in range(number_of_generation):
#     print('-- Generation {0} --'.format(generation + 1))

#     offspring = toolbox.select(pop, len(pop))
#     offspring = list(map(toolbox.clone, offspring))

#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < probability_of_crossover:
#             toolbox.mate(child1, child2)
#             del child1.fitness.values
#             del child2.fitness.values

#     for mutant in offspring:
#         if random.random() < probability_of_mutation:
#             toolbox.mutate(mutant)
#             del mutant.fitness.values

#     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#     fitnesses = map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit

#     print('  Evaluated %i individuals' % len(invalid_ind))

#     pop[:] = offspring
#     fits = [ind.fitness.values[0] for ind in pop]

#     length = len(pop)
#     mean = sum(fits) / length
#     sum2 = sum(x * x for x in fits)
#     std = abs(sum2 / length - mean ** 2) ** 0.5

#     print('  Min %s' % min(fits))
#     print('  Max %s' % max(fits))
#     print('  Avg %s' % mean)
#     print('  Std %s' % std)

# print('-- End of (successful) evolution --')

# best_individual = tools.selBest(pop, 1)[0]
# best_individual_array = np.array(best_individual)
# selected_X_variable_numbers = np.where(best_individual_array > threshold_of_variable_selection)[0]
# print('Selected variables : %s, %s' % (selected_X_variable_numbers * 2 + 700, best_individual.fitness.values))

import random
import numpy as np

from deap import base
from deap import creator
from deap import tools
from sklearn import datasets
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression


class Ga:
	def __init__(self, x, y, n_variables_wanted, n_sets = 1, n_cross_gene = 1):
		self.x = x
		self.n = x.shape[0]
		self.p = x.shape[1]
		self.y = y
		self.sts = n_sets
		self.p_want = n_variables_wanted
		self.n_cross_gene = n_cross_gene

	def _first_parents(self):
		parents = []
		for i in range(self.n):
			idx = np.random.randint(self.p, size = self.p_want)
			print(idx * 2 + 700)
			parents.append(self.x[i, idx])
		print(parents)
		parents = np.array_split(np.array(parents), self.sts)
		return parents	#list

	def fit(self, parents, rms_list):
		keep_idx = [a for a in range(len(rms_list)) if a != rms_list.index(max(rms_list))]
		return parents[keep_idx]

	def crossover(self, parents):
		# print(parents)
		new_set = []
		print(len(parents))
		print(parents)
		for _ in range(int(parents[-1].shape[0] / 2)):
			a = np.random.randint(len(parents), size = 2) #which two sample sets to gen. child
			b = np.random.randint(parents[a[0]].shape[0], size = 1)	# which two parent samples to gen. child
			c = np.random.randint(parents[a[1]].shape[0], size = 1)
			d = np.random.randint(low = 0 + self.n_cross_gene, high = parents[a[0]].shape[1] - self.n_cross_gene, size = 1)
			e = list(range(d, d + 10, 1))
			off1 = parents[a[0]][b]
			off2 = parents[a[0]][c]
			off1[e] = parents[a[0]][c, e]
			off2[e] = parents[a[0]][b, e]
			new_set.append(off1, off2)


		offspring = parents.append(new_set)
		# print(offspring)

		return offspring

	def mutate(self):


		return None

def main():
	X = np.random.rand(10, 10)
	Y = np.random.rand(10, 1)
	print(X)

	ga = Ga(X, Y, 3, 5)
	# print(ga._first_parents())
	ga.crossover(ga._first_parents())

	return None

if __name__ == '__main__':
	main()