from os.path import abspath, dirname, join
from time import process_time

import pandas as pd
import pygad
from matplotlib import pyplot as plt

from auxiliar import (
    create_confusion_matrix,
    exception_handler,
    finished_time,
    format,
    plot_confusion_matrix,
)
from chromossome import Chromossome, Gene

# ---- GA CONSTANTS ----
# 50 generations + 50 sol per pop = 5 min de execucao
# 50 generations + 100 sol per pop = 15 min de execucao

# number of generations
NUM_GENERATIONS = 50  # 50
# number of solutions to be selected as parents
NUM_PARENTS_MATING = 5  # 5
# number of solutions (chromossomes)
SOL_PER_POP = 50  # 50
# sss | rws | rank | tournament
PARENT_SELECTION_TYPE = "rank"  # rank
# number of parents to keep in the current population
KEEP_PARENTS = 1  # 1
# single_point | two_points | uniform | scattered | a custom crossover function
CROSSOVER_TYPE = "uniform"  # uniform
# probability of crossover
CROSSOVER_PROBABILITY = 0.9  # 0.9
# random | swap | inversion | scramble | adaptive | a custom mutation function
MUTATION_TYPE = "random"  # random
# percentage of genes to mutate
MUTATION_PERCENT_GENES = 10  # 10
# gene limit values
# GENE_SPACE = {'low': 0, 'high': 10}
# -1 e 1 roda 10 vezes e olha os parametros >> aumenta o intervalo com base nos resultados


# @ exception_handler
def ga(chromossome, output_path):
    """function to execute the genetic algorithm"""
    print("Initiating GA...")
    ga_start_time = process_time()

    def fitness_func(solution, solution_idx):
        """pygad fitness function to give as a parameter"""
        chromossome.chromossomePredict(solution)
        chromossome._setConfusionMatrix()
        chromossome._setFitness()
        chromossome._setSolution(solution)

        solution_fitness = chromossome._getFitness()
        return solution_fitness

    def prepare_ga():
        """function to prepare ga parameters"""
        print("Preparing GA...")
        num_genes = len(chromossome._getGenes())
        fitness_function = fitness_func

        def on_generation(ga_instance):
            fit = chromossome._getFitness()
            # s = chromossome._getSolution()
            print(f"\n GENERATION - Fitness: {format(fit)}")

        ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            fitness_func=fitness_function,
            sol_per_pop=SOL_PER_POP,
            num_genes=num_genes,
            parent_selection_type=PARENT_SELECTION_TYPE,
            keep_parents=KEEP_PARENTS,
            crossover_type=CROSSOVER_TYPE,
            crossover_probability=CROSSOVER_PROBABILITY,
            on_generation=on_generation,
            mutation_type=MUTATION_TYPE,
            mutation_percent_genes=MUTATION_PERCENT_GENES,
            #    gene_space=GENE_SPACE # fitness ficou fixo..
        )

        return ga_instance

    # prepare ga instance
    ga_instance = prepare_ga()
    # run ga instance
    ga_instance.run()
    # get output parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # plot fitness function per generation
    ga_instance.plot_fitness(save_dir=output_path)

    finished_time(ga_start_time, "GA ALGORITHM")
    return solution, solution_fitness, solution_idx
