import pandas as pd
from math_func import math_func

N_MATH = 4  # dist * rsa


class Gene():
    '''Gene is one dataframe grouped by same mutation'''

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.df = df_value
        self.fitness = 0

    def __str__(self, show_df=False):
        print(f'\tGENE = {self.wild_aa} -> {self.new_aa}', end=' | ')
        print(f'fit = {self.fitness_mean} | {len(self.df)} mutations', end='')
        # print(f'\n\n{self.df}')
        return ''

    def _getDataFrame(self):
        return self.df

    def _getFitness(self):
        return self.fitness_mean

    def _setFitness(self):
        self.fitness = self.df['fitness'].mean()

    def getYTrue(self):
        return self.df['severity'].values

    def getYPred(self):
        return self.df['fitness_discretized'].values

    def calculateFitness(self, s):
        def fitness(x):
            fitness = math_func[N_MATH](x=x, s=s)
            fitness = fitness
            return fitness

        self.df['fitness'] = self.df['rsa'].apply(fitness)

    def normalizeFitness(self):
        fitness = self.df['fitness'].values.copy()
        min, max = fitness.min(), fitness.max()
        if max - min != 0:
            fitness_normalized = [(v - min) / (max - min) for v in fitness]
        else:
            fitness_normalized = -1

        self.df['fitness_normalized'] = fitness_normalized

    def discretizeFitness(self, lb=0.33, ub=0.66):
        def discretize(value):
            if value >= 0 and value < lb:
                return 'Mild'  # 0
            elif value >= lb and value < ub:
                return 'Moderate'  # 1
            elif value >= ub and value <= 1:
                return 'Severe'  # 2
            else:
                return 'Moderate'  # discretize error: classify as moderate

        self.df['fitness_discretized'] = self.df['fitness_normalized'].apply(
            discretize)

    def genePredict(self, s):
        self.calculateFitness(s=s)
        self.normalizeFitness()
        self.discretizeFitness(lb=0.33, ub=0.66)
        self._setFitness()


class Chromossome(Gene):
    '''Chromossome is several genes with different mutations'''

    def __init__(self, genes=[]):
        self.genes = genes
        self.fitness = 0
        self.cm = 0

    def __str__(self):
        i_max = 2
        i = 1
        print(f'CHROMOSSOME with {len(self.genes)} genes', end=' - ')
        print(f'Printing only {i_max} genes\n')
        for gene in self.genes:
            if i <= i_max:
                print(f'{gene}')
            i += 1
        if not isinstance(self.cm, int):
            print(f'\nConfusion Matrix: \n{self.cm}')
        return ''

    def _getFitness(self):
        return self.fitness_mean

    def _setFitness(self):
        self.fitness = self.df['fitness'].mean()

    def _getGenes(self):
        return self.genes

    def _getConfusionMatrix(self):
        return self.cm

    def chromossomePredict(self, solution):
        chr_size = len(self.genes)
        sol_size = len(solution)

        if chr_size == sol_size:
            # print('Solution and Chromossome have the same size')
            for index, gene in enumerate(self.genes):
                s = solution[index]
                gene.genePredict(s)
        else:
            print('Solution and Chromossome DONT have the same size!!!!')

    def mergeGenesDataframes(self):
        df = pd.DataFrame()
        for gene in self.genes:
            gene_df = gene._getDataFrame()
            df = pd.concat([df, gene_df])

        return df


class Population():
    def __init__(self, size=None):
        pass


class HemaGA():
    def __init__(self, df=None, initial_pop=None, n_generations=None, n_math=None):
        pass
