import pandas as pd

from auxiliar import create_confusion_matrix, math_func
from gene import Gene


class Chromossome(Gene):
    """Chromossome is several genes with different mutations"""

    N_MATH = 4  # dist * rsa

    def __init__(self, genes=[]):
        self.genes = genes
        self.cm = 0
        self.fitness = 0
        self.solution = []

    def __str__(self):
        return f"CHROMOSSOME has {len(self.genes)} genes"

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
        y_true = merged_df["severity"].values
        y_pred = merged_df["fitness_discretized"].values

        self.cm = create_confusion_matrix(y_true, y_pred, normalize="true", plot=plot)

    def chromossomePredict(self, solution):
        chr_size = len(self.genes)
        sol_size = len(solution)

        if chr_size == sol_size:
            # print('Solution and Chromossome have the same size')
            for index, gene in enumerate(self.genes):
                s = solution[index]
                gene.genePredict(s)
        else:
            print("Solution and Chromossome DONT have the same size!!!!")

    def mergeGenesDataframes(self):
        df = pd.DataFrame()
        for gene in self.genes:
            gene_df = gene._getDataFrame()
            df = pd.concat([df, gene_df])

        return df
