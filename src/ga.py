import numpy as np

import time
import pygad
import numpy as np
import pandas as pd
from os.path import abspath, join, dirname
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score

from math_func import math_func
from auxiliar import discretize_to_severity, min_max_normalization, sev2int
from auxiliar import sum_diagonal_cf_matrix
from auxiliar import exception_handler, math_func_to_string
from preprocessing import get_initial_df, filter_unique_mutations, read_aa_dm_file, read_pm_file, read_rsa_file


from main import df_unique_mut, df
# ------------------------ Genetic Algorithm ------------------------------------- #


@ exception_handler
def get_initial_pop(data, size):
    '''get initial population for the ga'''
    return [np.array(data.dist_aa.values)] * size


@ exception_handler
def get_x_and_y(data):
    '''get x and y for prediction'''
    X = np.array(data.rsa.values)
    # Y = np.array([sev_to_int[i] for i in list(data.severity.values)])
    Y = data.sev.values
    return X, Y


# random state to reproduce sklearn classifiers
RANDOM_STATE = 42
# number of generations
NUM_GENERATIONS = 50
# math function number
N_MATH_FUNCTION = 4  # 4 is x*s

Y_TRUE_TOTAL = np.array([sev2int[i] for i in list(df.severity.values)])
X, Y_TRUE_UNIQUE = get_x_and_y(df_unique_mut)
# print(f'\nX = {X[:5]}')
# print(f'y = {Y_TRUE[:5]} len:{len(Y_TRUE)}')

INITIAL_POP = get_initial_pop(df_unique_mut, 50)
# print(f'first element of initial pop = {INITIAL_POP[:1][:5]}')
# print(f'pop size = {len(INITIAL_POP)}')


@ exception_handler
def ga(n=4, plot=False):
    '''function to execute the genetic algorithm'''

    def chromosome_fitness(solution):
        '''function to calculate chromossome fitness'''
        # originally is equal to this >> dist_aa = row.dist_aa
        df_unique_mut['solution'] = solution

        def gene_fitness(row):
            '''function to calculate gene fitness'''
            s = row.solution
            rsa_vector = row.rsa
            y_true = row.sev

            fitness_vector = []
            for rsa in rsa_vector:
                # calculate the fitness value for each rsa with same distance (s)
                fitness = math_func[n](x=rsa, s=s)
                fitness_vector.append(fitness)

            # normalize the chromossome fitness between 0 and 1
            fitness_vector_normalized = min_max_normalization(fitness_vector)

            # discretize the chromossome fitness
            fitness_vector_discretized = discretize_to_severity(
                fitness_vector_normalized)

            y_pred = fitness_vector_discretized

            acc = accuracy_score(y_true, y_pred)
            # matriz de distancias
            hits = sum(1 for s in y_pred if s == y_true)
            # print(f'accuracy = {acc} - hits = {hits}')
            return acc

            # confusion matrix normalized by row
            # cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

            # plot confusion matrix as heatmap
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
            # ax.set_title(f'Mutation: {row.wild_aa} -> {row.new_aa}')
            # ax.set_xlabel('Predicted Hemophilia Severity')
            # ax.set_ylabel('Actual Hemophilia Severity')
            # plt.show()
            # print(cf_matrix)

            # solution fitness - sum of the correct cases
            solution_fitness = sum_diagonal_cf_matrix(cf_matrix)
            return solution_fitness

        df_unique_mut['fitness'] = df_unique_mut.apply(gene_fitness, axis=1)
        gene_fitness_vector = df_unique_mut['fitness'].values

        return gene_fitness_vector

    # def predict_ga(solution):
    #     '''function to calculate and discretize the chromossome fitness'''

    #     # calculate the chromossome fitness
    #     c_fitness = chromosome_fitness(solution)
    #     # normalize the chromossome fitness between 0 and 1
    #     c_fitness_normalized = min_max_normalization(c_fitness)
    #     # discretize the chromossome fitness
    #     c_fitness_discretized = discretize_to_severity(c_fitness_normalized)

    #     return c_fitness_discretized

    # def fitness_func(solution, solution_idx):
    #     '''pygad fitness function to give as a parameter'''

    #     # y predict is the chromossome fitness discretized
    #     y_pred = predict_ga(solution)
    #     # confusion matrix normalized by row
    #     cf_matrix = confusion_matrix(Y_TRUE_UNIQUE, y_pred, normalize='true')
    #     # solution fitness - sum of the correct cases
    #     solution_fitness = sum_diagonal_cf_matrix(cf_matrix)

    #    return solution_fitness

    def fitness_func(solution, solution_idx):
        '''pygad fitness function to give as a parameter'''

        # vector of accuracy
        c_fitness = chromosome_fitness(solution)
        solution_fitness = np.mean(c_fitness)

        # print(c_fitness, solution_fitness)
        return solution_fitness

    def prepare_ga():
        '''function to prepare ga parameters'''

        # parameters
        fitness_function = fitness_func

        num_generations = NUM_GENERATIONS
        # number of solutions to be selected as parents
        num_parents_mating = 5

        # sol_per_pop and num_genes are None when given initial population
        initial_population = INITIAL_POP
        # number of solutions (i.e. chromossomes) within the population
        # sol_per_pop = 50  # 50 fica legal!
        # num_genes = len(X)  # NUM_GENES
        # used to specify the possible values for each gene in case the user wants to restrict the gene values.
        # gene_space = {'low': 0, 'high': 1}

        # sss = steady state selection; rws = roulette wheel selection; rank; tournament;
        parent_selection_type = "rank"  # rank fica legal!
        # number of parents to keep in the current population
        keep_parents = 1  # keep parents = 1 it's important because improve results

        # "single_point", "two_points", "uniform", "scattered" or a custom crossover function
        crossover_type = "uniform"  # uniform fica legal!

        # "random", "swap", "inversion", "scramble", "adaptive" or custom mutation function
        mutation_type = "random"
        # percentage of genes to mutate
        mutation_percent_genes = 10

        # instance of pygad.GA using parameters defined
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               initial_population=initial_population,
                               #    sol_per_pop=sol_per_pop,
                               #    num_genes=num_genes,
                               #    gene_space=gene_space,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)

        return ga_instance

    # prepare ga instance
    ga_instance = prepare_ga()

    # run ga instance
    ga_instance.run()

    # get output parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # plot fitness function per generation
    ga_instance.plot_fitness()

    return solution, solution_idx, solution_fitness

    # # ga prediction using best solution
    # best_y_pred = predict_ga(solution)

    # # create confusion_matrix and normalize = true uses row to normalize
    # best_cf_matrix = confusion_matrix(
    #     Y_TRUE_UNIQUE, best_y_pred, normalize='true')

    # def print_ga():
    #     '''function to print and plot results of the genetic algorithm'''

    #     # print solution and fitness value
    #     # print(f'\nArray of best solution : {solution}: {(len(solution))}')
    #     # print(f'\nFitness value of the best solution = {solution_fitness}')

    #     # plot confusion matrix labeled
    #     plot_cf_matrix(best_cf_matrix)

    #     # plot fitness function per generation
    #     ga_instance.plot_fitness()

    # if plot:
    #     print_ga()

    # return best_y_pred, solution, solution_fitness


@ exception_handler
def get_number_best_math_function(plot=False):
    '''function to get n of the math function used in the ga'''
    def loop_ga():
        '''execute ga with every math function defined initially'''
        dict_solutions = {}
        for n in range(len(math_func)):
            # convert math func to string
            str_func = math_func_to_string(math_func[n])
            # execute ga with determined math func
            _, solution, solution_fitness = ga(n, plot)
            # keep best solution information as dict
            dict_solutions[n] = {
                'math_func': str_func,
                'best_solution': solution,
                'best_solution_fitness': solution_fitness
            }

        df_solutions = pd.DataFrame(dict_solutions).T
        return df_solutions

    # execute ga using multiple math functions
    df_sol = loop_ga()
    # sort by solution fitness
    df_sol.sort_values('best_solution_fitness', ascending=False, inplace=True)
    print(df_sol)
    # best math function
    math_function = df_sol.math_func.iloc[0]
    n = df_sol.index[0]

    return math_function, n
