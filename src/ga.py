import numpy as np
import pygad
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from auxiliar import math_funcs


class GA:
    """GA class"""

    def __init__(self, rsa, sev_true):
        self.rsa = rsa
        self.sev_true = sev_true

    def __str__(self):
        return f"Fitness={self.fitness}"

    def calculate_fitness(self, solution, n):
        fitness = []
        for s, r in zip(solution, self.rsa):
            f = math_funcs[n](s, r)
            fitness.append(f)

        self.fitness = fitness

    def normalize_fitness(self):
        f_min = min(self.fitness)
        f_max = max(self.fitness)

        for i, f in enumerate(self.fitness):
            self.fitness[i] = (f - f_min) / (f_max - f_min)

    def discretize_fitness(self, lb=0.33, ub=0.66):
        y_pred = []
        lb = np.quantile(self.fitness, lb)
        ub = np.quantile(self.fitness, ub)
        for f in self.fitness:
            if 0 < f <= lb:
                y_pred.append("Mild")
            elif lb < f <= ub:
                y_pred.append("Moderate")
            elif ub < f <= 1:
                y_pred.append("Severe")
            else:
                y_pred.append("None")

        self.sev_pred = y_pred

    def solution_fitness(self):
        hits = 0
        for y_pred, y_true in zip(self.sev_pred, self.sev_true):
            if y_pred == y_true:
                hits += 1

        self.hits = hits

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.sev_true, self.sev_pred, normalize="true")

        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Confusion Matrix normalized by row")
        plt.show()

    def pygad(self, n):
        def fitness_func(solution, solution_idx):
            self.calculate_fitness(solution, n)
            self.normalize_fitness()
            self.discretize_fitness()
            self.solution_fitness()

            return round((self.hits / len(self.rsa)) * 100, 2)

        fitness_function = fitness_func

        num_generations = 100
        num_parents_mating = 4

        sol_per_pop = 8
        num_genes = len(self.rsa)
        gene_space = list(range(1, 100, 1))

        init_range_low = -3
        init_range_high = 5

        parent_selection_type = "sss"
        keep_parents = 2

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 10

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
        )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Accuracy of the best solution = {solution_fitness}%")
        # ga_instance.plot_fitness()
        # print(np.unique(self.sev_pred))
        # self.plot_confusion_matrix()

        return solution, solution_fitness, solution_idx

        # print("Parameters of the best solution : {solution}".format(solution=solution))
        # print(
        #     "Fitness value of the best solution = {solution_fitness}".format(
        #         solution_fitness=solution_fitness
        #     )
        # )

        # prediction = numpy.sum(numpy.array(function_inputs) * solution)
        # print(
        #     "Predicted output based on the best solution : {prediction}".format(
        #         prediction=prediction
        #     )
        # )
