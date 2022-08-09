from os.path import abspath, join, dirname

from preprocessing import initial_df, filter_df_for_ga
from hemaGA import Gene

# from hemaGA import hemaGA, get_initial_pop, get_X, get_Y_true

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

    print('\n\n\t\tCLASSES!!!!\n\n')
    chromossome = df.groupby(['wild_aa', 'new_aa'])  # chromossomes

    for gene_k, gene_v in chromossome:
        g1 = Gene(gene_k, gene_v)
        g1.calculateFitness(1.5)
        g1.normalizeFitness()
        g1.discretizeFitness()
        print(g1)
        g1.confusionMatrix()
        break


# ''' Genetic Algorithm'''
# df_ga = filter_df_for_ga(df)

# POP_SIZE = 50
# NUM_GENS = 10
# N_MATH_FUNC = 4

# initial_pop = get_initial_pop(df_ga, POP_SIZE)
# y_true = get_Y_true(df_ga)
# X = get_X(df_ga)

# # n = n_best_math_func(plot=False)
# solution, solution_idx, solution_fitness = hemaGA(
#     df_ga, initial_pop, num_gens=NUM_GENS, n=N_MATH_FUNC, plot=False)

# cf_matrix = confusion_matrix(y_true, solution, normalize='true')
# plot_cf_matrix(cf_matrix)
