import numpy as np
import pygad

from auxiliar import exception_handler, finished_time
from chromosome import Chromosome


@exception_handler
class GA:
    """GA is a class to setup and execute pygad ga using my gene and chromosome classes"""

    # ---- GA CONSTANTS ----

    # number of generations
    NUM_GENERATIONS = 50
    # number of solutions to be selected as parents
    NUM_PARENTS_MATING = 5
    # number of solutions (chromossomes)
    SOL_PER_POP = 50
    # sss | rws | rank | tournament
    PARENT_SELECTION_TYPE = "rank"
    # number of parents to keep in the current population
    KEEP_PARENTS = 1
    # single_point | two_points | uniform | scattered | a custom crossover function
    CROSSOVER_TYPE = "uniform"
    # probability of crossover
    CROSSOVER_PROBABILITY = 0.9
    # random | swap | inversion | scramble | adaptive | a custom mutation function
    MUTATION_TYPE = "random"
    # percentage of genes to mutate
    MUTATION_PERCENT_GENES = 10  # 10
    # gene limit values
    # GENE_SPACE = {'low': 0, 'high': 10}

    def __init__(self, df):
        self.solution = Chromosome(df).genes
        self.chromosome = Chromosome(df)

    def __str__(self):
        print(self.best_solution)
        print(self.best_solution_idx)
        print(self.best_solution_fitness)
        return ""

    def on_generation(self):
        print(f"\tOn Generation ...")
        # print(self)

    def setup(self):
        """Function to setup GA parameters"""
        print("Setting up GA parameters...")

        def fitness_function(solution, solution_idx):
            """Fitness Function to execute in pygad"""
            print("Executing fitness function...")

            solution = self.solution
            fitness = self.chromosome.solution_calculation(solution)

            return fitness

        # setup GA instance
        self.ga_instance = pygad.GA(
            num_generations=self.NUM_GENERATIONS,
            num_parents_mating=self.NUM_PARENTS_MATING,
            sol_per_pop=self.SOL_PER_POP,
            parent_selection_type=self.PARENT_SELECTION_TYPE,
            keep_parents=self.KEEP_PARENTS,
            crossover_type=self.CROSSOVER_TYPE,
            crossover_probability=self.CROSSOVER_PROBABILITY,
            mutation_type=self.MUTATION_TYPE,
            mutation_percent_genes=self.MUTATION_PERCENT_GENES,
            on_generation=self.on_generation(),
            fitness_func=fitness_function,
            num_genes=len(self.solution),
        )

    def execute(self):
        # run GA
        self.ga_instance.run()

        # get best solution
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()

        self.best_solution = solution
        self.best_solution_idx = solution_idx
        self.best_solution_fitness = solution_fitness

        # plot fitness function per generation
        self.ga_instance.plot_fitness()
