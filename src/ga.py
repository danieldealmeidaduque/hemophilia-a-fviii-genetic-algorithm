from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from auxiliar import math_funcs


class GA:
    """GA class"""

    def __init__(self, rsa, dist, y_true, n_math_func=4):
        self.rsa = rsa
        self.dist = dist
        self.y_true = y_true
        self.y_pred = []
        self.fitness = []
        self.n_math_func = n_math_func
        self.hits = 0

    def __str__(self):
        # print(func2str(math_funcs[self.n_math_func]), end=" | ")
        return f"Dist={self.dist} - RSA={self.rsa} - Fitness={self.fitness}"

    def calculate_fitness(self):
        fitness_list = []
        for r, d in zip(self.rsa, self.dist):
            r = self.rsa
            d = self.dist
            n = self.n_math_func
            f = math_funcs[n](d, r)
            fitness_list.append(f)

        self.fitness = fitness_list

    def normalize_fitness(self):
        self.fitness = preprocessing.normalize(self.fitness)[0].tolist()

    def discretize_fitness(self, lb=0.33, ub=0.66):
        y_pred = []
        for f in self.fitness:
            if 0 < f <= lb:
                y_pred.append('Mild')
            elif lb < f <= ub:
                y_pred.append('Moderate')
            elif ub < f <= 1:
                y_pred.append('Severe')
            else:
                y_pred.append('None')

        self.y_pred = y_pred

    def solution_fitness(self):
        hits = 0
        for y_pred, y_true in zip(self.y_pred, self.y_true):
            if y_pred == y_true:
                hits += 1

        self.hits = hits

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred, normalize="true")
