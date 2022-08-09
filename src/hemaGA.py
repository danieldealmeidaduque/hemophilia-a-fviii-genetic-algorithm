import pygad
import numpy as np
from time import process_time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


from math_func import math_func
from auxiliar import discretize_to_severity, min_max_normalization, finished_time, sev2int
from auxiliar import sum_diagonal_cf_matrix, exception_handler, math_func2string
from auxiliar import format
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

n = 4  # dist * rsa


class Gene():
    '''Gene is a dataframe grouped by same mutation'''

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.fitness = 0
        self.df = df_value

    def __str__(self):
        print(f'\nGENE \n{self.wild_aa} -> {self.new_aa}')
        print(f'Fitness: {self.fitness}')
        print(self.df)
        return ''

    def calculateFitness(self, s):
        def fitness(x):
            fitness = math_func[n](x=x, s=s)
            fitness = fitness
            return fitness

        self.df['fitness'] = self.df['rsa'].apply(fitness)

    def normalizeFitness(self):
        fitness = self.df['fitness'].values.copy()
        min, max = fitness.min(), fitness.max()

        if max - min != 0:
            fitness_normalized = [(v - min) / (max - min) for v in fitness]

        self.df['fitness_normalized'] = fitness_normalized

    def discretizeFitness(self, lb=0.33, ub=0.66):
        def discretize(value):
            if value >= 0 and value < lb:
                return 'Mild'  # 0
            elif value >= lb and value < ub:
                return 'Moderate'  # 1
            elif value >= ub and value <= 1:
                return 'Severe'  # 2

        self.df['fitness_discretized'] = self.df['fitness_normalized'].apply(
            discretize)

    def confusionMatrix(self):
        y_true = self.df['severity']
        y_pred = self.df['fitness_discretized']
        cf_labels = ['Mild', 'Moderate', 'Severe']

        cf_matrix = confusion_matrix(
            y_true, y_pred, labels=cf_labels, normalize='true')

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix, display_labels=cf_labels)

        disp.plot()
        plt.show()

    '''def gene_fitness(row):
            #function to calculate gene fitness
            s = row.solution
            rsa_vector = row.rsa
            y_true = row.sev

            
            # normalize the chromossome fitness between 0 and 1
            fitness_vector_normalized = min_max_normalization(fitness_vector)

            # discretize the chromossome fitness
            fitness_vector_discretized = discretize_to_severity(
                fitness_vector_normalized)

            y_pred = fitness_vector_discretized

            acc = accuracy_score(y_true, y_pred)
            # matriz de distancias
            hits = sum(1 for s in y_pred if s == y_true)
            # print(f'accuracy = {acc} - hits = {hits}')
            return acc'''
    pass


class Chromossome():
    '''Chromossome is several genes with different mutations'''

    def __init__(self, df):
        self.chromossome = df
        self.fitness = 0

    def calculateFitness(self, dist_aa):

        for gene in range(self.chromossome):
            f_value = math_func[n]()

    pass


class Population():
    def __init__(self, size=None):
        pass


class HemaGA():
    def __init__(self, df=None, initial_pop=None, n_generations=None, n_math=None):
        pass
