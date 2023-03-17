import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from gene import Gene


class Chromossome(Gene):
    """Chromossome is a list of genes"""

    def __init__(self, df):
        # TEMPORARY BECAUSE OF DISTANCE NaN
        df = df[df["HGVS New Amino Acid"] != "*"]

        self.genes = df.apply(lambda row: Gene(row), axis=1)
        self.solution = None
        self.solution_idx = None
        self.solutio_fitness = None

    def __str__(self, n=5):
        print(f"Chromossome has {len(self.genes)} genes")
        for g in self.genes.iloc[:n]:
            print(g)
        return ""

    def create_confusion_matrix(self):
        y_true = [g.sev for g in self.genes]
        y_pred = [g.fitness_discretize().fitness for g in self.genes]

        cm = confusion_matrix(y_true, y_pred, normalize="true", plot=True)
        return cm

    def fitness_calculate(self):
        for g in self.genes:
            g.fitness_calculate()

    def fitness_discretize(self):
        for g in self.genes:
            g.fitness_discretize()

    def fitness_normalize(self):
        fitness_list = [g.fitness for g in self.genes]
        fitness_list_normalized = preprocessing.normalize([fitness_list])[0]

        for index, g in enumerate(self.genes):
            g.fitness = fitness_list_normalized[index]
