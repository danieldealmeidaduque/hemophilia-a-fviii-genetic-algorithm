import pygad
from time import process_time
from matplotlib import pyplot as plt
from os.path import abspath, join, dirname
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocessing import initial_df
from hemaGA import Gene, Chromossome
from auxiliar import finished_time, exception_handler

# ---- GA CONSTANTS ----
# 50 generations + 50 sol per pop = 5 min de execucao

# number of generations
NUM_GENERATIONS = 10  # 50
# number of solutions to be selected as parents
NUM_PARENTS_MATING = 5  # 5
# number of solutions (chromossomes)
SOL_PER_POP = 50  # 50
# sss | rws | rank | tournament
PARENT_SELECTION_TYPE = 'rank'  # rank
# number of parents to keep in the current population
KEEP_PARENTS = 1  # 1
# single_point | two_points | uniform | scattered | a custom crossover function
CROSSOVER_TYPE = 'uniform'  # uniform
# probability of crossover
CROSSOVER_PROBABILITY = 0.9
# random | swap | inversion | scramble | adaptive | a custom mutation function
MUTATION_TYPE = 'random'  # random
# percentage of genes to mutate
MUTATION_PERCENT_GENES = 10  # 10


@ exception_handler
def create_confusion_matrix(y_true, y_pred, normalize=None, plot=False):
    labels = ['Mild', 'Moderate', 'Severe']
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if plot:
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix normalized by row')
        plt.show()

    return cm


@ exception_handler
def ga(chromossome):
    '''function to execute the genetic algorithm'''
    ga_start_time = process_time()

    def fitness_func(solution, solution_idx):
        '''pygad fitness function to give as a parameter'''
        chromossome.chromossomePredict(solution)
        merged_df = chromossome.mergeGenesDataframes()
        y_true = merged_df['severity'].values
        y_pred = merged_df['fitness_discretized'].values

        cm = create_confusion_matrix(
            y_true, y_pred, normalize='all', plot=False)
        solution_fitness = cm[0][0] + cm[1][1] + cm[2][2]

        # print(f'{solution_fitness}', end=' - ')
        return solution_fitness

    def prepare_ga():
        '''function to prepare ga parameters'''
        num_genes = len(chromossome._getGenes())
        fitness_function = fitness_func

        def on_generation(ga_instance):
            print(chromossome.mergeGenesDataframes())
            print(f'\nNEW GENERATION')

        ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
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
    return solution, solution_fitness, solution_idx


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

    df_grouped = df.groupby(['wild_aa', 'new_aa'])  # is the chromossome
    genes = [Gene(k, v) for k, v in df_grouped]  # all genes
    chromossome = Chromossome(genes)  # chromossome class

    solution, solution_fitness, solution_idx = ga(chromossome)
    print(solution_fitness)
    print(solution_idx)
    print(solution)
