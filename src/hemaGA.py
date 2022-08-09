import pygad
import numpy as np
from time import process_time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


from math_func import math_func
from auxiliar import discretize_to_severity, min_max_normalization, finished_time, sev2int
from auxiliar import sum_diagonal_cf_matrix, exception_handler, math_func2string
from auxiliar import format, scores, create_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

n = 4  # dist * rsa


class Gene():
    '''Gene is one dataframe grouped by same mutation'''

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.df = df_value
        self.score = 0
        self.cm = 0

    def __str__(self):
        print(f'\tGENE = {self.wild_aa} -> {self.new_aa}', end=' | ')
        print(f'{len(self.df)} mutations', end='')
        if not isinstance(self.cm, int):
            print(f'Confusion Matrix: \n{self.cm}')
        # print(self.df)
        return ''

    def getYTrue(self):
        return self.df['severity']

    def getYPred(self):
        return self.df['fitness_discretized']

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

    def predictionScore(self):
        y_true = self.df['severity']
        y_pred = self.df['fitness_discretized']

        self.score = scores(y_true, y_pred)

    def confusionMatrix(self, normalize=None, plot=False):
        y_true = self.df['severity']
        y_pred = self.df['fitness_discretized']

        self.cm = create_confusion_matrix(y_true, y_pred, normalize, plot)


class Chromossome():
    '''Chromossome is several genes with different mutations'''

    def __init__(self, chromosome=[]):
        self.chromossome = chromosome
        self.cm = 0

    def __str__(self, i_max=5):
        print(f'CHROMOSSOME with {len(self.chromossome)} genes', end=' - ')
        print(f'Printing only {i_max} genes')
        i = 1
        for gene in self.chromossome:
            if i <= i_max:
                print(f'{gene}')
            i += 1
        if not isinstance(self.cm, int):
            print(f'Confusion Matrix: \n{self.cm}')
        return ''

    def _getConfusionMatrix(self):
        return self.cm

    def calculateFitness(self, solution):
        sol_len = len(solution)
        chr_len = len(self.chromosome)
        print(sol_len, chr_len)

    def addGene(self, gene):
        self.chromossome.append(gene)

    def confusionMatrix(self, normalize=None, plot=False):
        y_true, y_pred = [], []
        for gene in self.chromossome:
            y_true.append(gene._getYTrue())
            y_pred.append(gene._getYPred())

        self.cm = create_confusion_matrix(y_true, y_pred, normalize, plot)


class Population():
    def __init__(self, size=None):
        pass


class HemaGA():
    def __init__(self, df=None, initial_pop=None, n_generations=None, n_math=None):
        pass
