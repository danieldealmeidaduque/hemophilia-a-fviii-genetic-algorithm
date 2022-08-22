import pandas as pd
from math_func import math_func
from auxiliar import create_confusion_matrix

N_MATH = 4  # dist * rsa


class Gene():
    '''Gene is one dataframe grouped by same mutation'''

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.df = df_value
        self.fitness = 0

    def __str__(self, show_df=False):
        print(f'\tGENE = {self.wild_aa} -> {self.new_aa}', end=' | ')
        print(f'{len(self.df)} mutations | fit = {self.fitness_mean}', end='')
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

    def discretizeFitness(self, lb=0.44, ub=0.55):
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
        self.cm = 0
        self.fitness = 0
        self.solution = []

    def __str__(self):
        return f'CHROMOSSOME has {len(self.genes)} genes'

    def _getGenes(self):
        return self.genes

    def _getSolution(self):
        return self.solution

    def _setSolution(self, solution):
        self.solution = solution

    def _getFitness(self):
        return self.fitness

    def _setFitness(self):
        self.fitness = self.cm[0][0] + self.cm[1][1] + self.cm[2][2]

    def _getConfusionMatrix(self):
        return self.cm

    def _setConfusionMatrix(self, plot=False):
        merged_df = self.mergeGenesDataframes()
        y_true = merged_df['severity'].values
        y_pred = merged_df['fitness_discretized'].values

        self.cm = create_confusion_matrix(
            y_true, y_pred, normalize='true', plot=plot)

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
