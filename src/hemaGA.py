import pygad
import numpy as np
from time import process_time
from sklearn.metrics import accuracy_score

from math_func import math_func
from auxiliar import discretize_to_severity, min_max_normalization, finished_time, sev2int
from auxiliar import sum_diagonal_cf_matrix, exception_handler, math_func2string

n = 4  # dist * rsa


class Gene():
    '''Gene is a dataframe grouped by same mutation'''

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.df = df_value
        self.fitness = 0

    def __str__(self):
        print(f'Gene: {self.wild_aa} -> {self.new_aa}', end=' - ')
        print(f'Fitness: {self.fitness}\n')
        print(self.df)
        return ''

    def calculateFitness(self, dist_aa):

        for gene in range(self.chromossome):
            f_value = math_func[n]()

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
