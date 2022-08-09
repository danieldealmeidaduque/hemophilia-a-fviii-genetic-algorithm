import pygad
import numpy as np
from time import process_time
from os.path import abspath, join, dirname

from auxiliar import finished_time
from preprocessing import initial_df
from hemaGA import Gene, Chromossome

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
CROSSOVER_PROBABILITY = 0.5
# random | swap | inversion | scramble | adaptive | a custom mutation function
MUTATION_TYPE = "random"
# percentage of genes to mutate
MUTATION_PERCENT_GENES = 10

if __name__ == '__main__':
    '''Get initial data and execute genetic algorithm'''

    input_dir = abspath(join(dirname(__file__), '..', 'datasets'))

    input_pm_file = 'FVIII_point_mutations_v1.csv'
    input_pm_path = join(input_dir, input_pm_file)

    input_dm_file = 'Supplementary_Table_npj_paper.xlsx'
    input_dm_path = join(input_dir, input_dm_file)

    input_rsa_file = 'Relative_Surf_Area_2R7E_v2.csv'
    input_rsa_path = join(input_dir, input_rsa_file)

    df = initial_df(input_pm_path, input_dm_path, input_rsa_path)

    # ---- GENETIC ALGORITHM ----

    print('\n\n')
    chromossome_df = df.groupby(['wild_aa', 'new_aa'])
    num_genes = len(chromossome_df)
    chromossome_vector = [Gene(k, v) for k, v in chromossome_df]
    chromossome_class = Chromossome(chromossome_vector)
    print(chromossome_class)

    # for gene_k, gene_v in chromossome:
    #     g1 = Gene(gene_k, gene_v)
    #     g1.calculateFitness(1.5)
    #     g1.normalizeFitness()
    #     g1.discretizeFitness()
    #     cm = g1.confusionMatrix()
    #     s = g1.predictionScore()
    #     y_true = g1.getYTrue()
    #     y_pred = g1.getYPred()


def ga(df):
    '''function to execute the genetic algorithm'''
    ga_start_time = process_time()
    # chromossome_df = df.groupby(['wild_aa', 'new_aa'])
    # num_genes = len(chromossome_df)
    # chromossome_vector = [Gene(k, v) for k, v in chromossome_df]
    # chromossome_class = Chromossome(chromossome_vector)

    def fitness_func(solution, solution_idx):
        '''pygad fitness function to give as a parameter'''

        # c_fitness = chromosome_fitness(solution)
        # solution_fitness = np.mean(c_fitness)
        return solution_fitness

    def prepare_ga():
        '''function to prepare ga parameters'''

        fitness_function = fitness_func

        ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                               num_parents_mating=NUM_PARENTS_MATING,
                               fitness_func=fitness_function,
                               sol_per_pop=SOL_PER_POP,
                               num_genes=num_genes,
                               parent_selection_type=PARENT_SELECTION_TYPE,
                               keep_parents=KEEP_PARENTS,
                               crossover_type=CROSSOVER_TYPE,
                               crossover_probability=CROSSOVER_PROBABILITY,
                               mutation_type=MUTATION_TYPE,
                               mutation_percent_genes=MUTATION_PERCENT_GENES)

        return ga_instance

    # prepare ga instance
    ga_instance = prepare_ga()
    # run ga instance
    ga_instance.run()
    # get output parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # plot fitness function per generation
    ga_instance.plot_fitness()

    finished_time(ga_start_time, 'GA ALGORITHM')
    return solution, solution_idx, solution_fitness
