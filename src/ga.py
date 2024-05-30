import numpy as np
import pygad
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from auxiliar import math_funcs


class GA:
    """Genetic Algorithm class for optimizing a solution based on fitness calculated using custom mathematical functions."""

    def __init__(self, rsa, sev_true):
        """
        Initialize the GA class with the given RSA and severity ground truth.

        :param rsa: List of RSA values
        :param sev_true: List of true severity labels
        """
        self.rsa = rsa
        self.sev_true = sev_true
        self.fitness = []
        self.sev_pred = []
        self.hits = 0

    def __str__(self):
        return f"Fitness={self.fitness}"

    def calculate_fitness(self, solution, n):
        """
        Calculate the fitness of a given solution.

        :param solution: List of solutions
        :param n: Index to select the mathematical function
        """
        try:
            self.fitness = [math_funcs[n](s, r) for s, r in zip(solution, self.rsa)]
        except Exception as e:
            print(f"Error in calculating fitness: {e}")
            self.fitness = [0] * len(solution)

    def normalize_fitness(self):
        """Normalize the fitness values to the range [0, 1]."""
        try:
            f_min, f_max = min(self.fitness), max(self.fitness)
            if f_max == f_min:
                self.fitness = [0] * len(self.fitness)
            else:
                self.fitness = [(f - f_min) / (f_max - f_min) for f in self.fitness]
        except Exception as e:
            print(f"Error in normalizing fitness: {e}")
            self.fitness = [0] * len(self.fitness)

    def discretize_fitness(self, lb=0.33, ub=0.66):
        """
        Discretize the fitness values into severity categories.

        :param lb: Lower bound quantile for categorization
        :param ub: Upper bound quantile for categorization
        """
        try:
            lb_value = np.quantile(self.fitness, lb)
            ub_value = np.quantile(self.fitness, ub)
            self.sev_pred = [
                "Mild" if 0 < f <= lb_value else
                "Moderate" if lb_value < f <= ub_value else
                "Severe" if ub_value < f <= 1 else
                "None"
                for f in self.fitness
            ]
        except Exception as e:
            print(f"Error in discretizing fitness: {e}")
            self.sev_pred = ["None"] * len(self.fitness)

    def solution_fitness(self):
        """Calculate the number of correct predictions."""
        try:
            self.hits = sum(1 for y_pred, y_true in zip(self.sev_pred, self.sev_true) if y_pred == y_true)
        except Exception as e:
            print(f"Error in calculating solution fitness: {e}")
            self.hits = 0

    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        try:
            cm = confusion_matrix(self.sev_true, self.sev_pred, normalize="true")
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.title("Confusion Matrix normalized by row")
            plt.show()
        except Exception as e:
            print(f"Error in plotting confusion matrix: {e}")

    def run_pygad(self, n):
        """
        Run the genetic algorithm to find the best solution.

        :param n: Index to select the mathematical function
        :return: Best solution, its fitness, and index
        """
        def fitness_func(ga_instance, solution, solution_idx):
            self.calculate_fitness(solution, n)
            self.normalize_fitness()
            self.discretize_fitness()
            self.solution_fitness()
            return round((self.hits / len(self.rsa)) * 100, 2)

        ga_instance = pygad.GA(
            num_generations=100,
            num_parents_mating=4,
            fitness_func=fitness_func,
            sol_per_pop=16,
            num_genes=len(self.rsa),
            gene_space=list(range(1, 100, 1)),
            init_range_low=-3,
            init_range_high=5,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            parallel_processing=("thread", 8)
        )

        try:
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            print(f"Accuracy of the best solution = {solution_fitness}%")
        except Exception as e:
            print(f"Error running GA: {e}")
            solution, solution_fitness, solution_idx = None, 0, -1

        return solution, solution_fitness, solution_idx
