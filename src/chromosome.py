import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from gene import Gene


class Chromosome(Gene):
    """chromosome is a list of genes"""

    def __init__(self, df):
        # TEMPORARY BECAUSE OF DISTANCE NaN
        df = df[df["HGVS New Amino Acid"] != "*"]

        self.genes = df.apply(lambda row: Gene(row), axis=1)
        self.solution = None
        self.solution_idx = None
        self.solution_fitness = None

    def __str__(self, n=5):
        print(f"chromosome has {len(self.genes)} genes")
        for g in self.genes.iloc[:n]:
            print(g)
        return f"{self.solution_fitness=}\n"

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

    def fitness_solution(self):
        f = 0
        for g in self.genes:
            print(g.sev_pred, g.sev_true)
            if g.sev_pred == g.sev_true:
                f += 1
        self.solution_fitness = f

    def solution_calculation(self, solution):
        # redefine the distance as solution[index]
        for index, gene in enumerate(self.genes):
            gene.dist = solution[index]

        # calculate fitness with pygad solution as new distance
        self.fitness_calculate()
        self.fitness_normalize()
        self.fitness_discretize()
        self.fitness_solution()

        return self.solution_fitness
