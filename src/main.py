# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from os.path import abspath, join, dirname

from ga import ga, get_number_best_math_function
from preprocessing import get_initial_df, filter_unique_mutations
from auxiliar import scores, confusion_matrix, plot_cf_matrix, print_finished_time, dummy_clf_scores

# ------------------------ Input Diretory and Files ------------------------------ #

# diretory
input_dir = abspath(join(dirname(__file__), '..', 'datasets'))

# input file - point mutations (pm)
pm_file = 'FVIII_point_mutations_v1.csv'
pm_path = join(input_dir, pm_file)

# input file - amino acids distance matrix (aa_dm)
aa_dm_file = 'Supplementary_Table_npj_paper.xlsx'
aa_dm_path = join(input_dir, aa_dm_file)

# input file - relative surface area (rsa)
rsa_file = 'Relative_Surf_Area_2R7E_v2.csv'
rsa_path = join(input_dir, rsa_file)

# dataframe with all informations
df = get_initial_df(aa_dm_path, rsa_path, pm_path)
print(f'\n{df}')

# dataframe filtered to use in ga
df_unique_mut = filter_unique_mutations(df)
print(f'df_filtered:\n\n{df_unique_mut}')


if __name__ == '__main__':
    # ------------------ Genetic Algorithm -----------------------------
    ga_start_time = time.time()

    # execute ga with several math functions to find the best one
    s, n = get_number_best_math_function(plot=False)
    print(f'\n{s} - function number {n} in dictionary of mathematical functions')

    # execute ga with an especific math function
    solution, solution_idx, solution_fitness = ga(n=4, plot=False)

    # create confusion_matrix and normalize = true uses row to normalize
    # best_cf_matrix = confusion_matrix(y_true, solution, normalize='true') # y_true??
    # plot_cf_matrix(best_cf_matrix)

    print_finished_time(ga_start_time, 'GA ALGORITHM')

    # ------------------ Recreating Distance Matrix with GA solution ---

    # ga solution to distance matrix
    df_unique_mut['dist_aa'] = solution
    # index and column names
    aa_list = np.unique(df_unique_mut['wild_aa'].values.tolist())
    # empty distance matrix labeled
    dm = pd.DataFrame(index=aa_list, columns=aa_list)

    # function to insert an amino acid distance in distance matrix
    def insert_aa_in_dm(row):
        dm.at[row.wild_aa, row.new_aa] = round(row.dist_aa, 2)

    df_unique_mut.apply(insert_aa_in_dm, axis=1)

    print(df.describe())
    print(df_unique_mut.describe())
    print(dm)

    # # ------------------ Other Classifiers and Scores ------------------
    # dict_scores = {}
    # dict_scores = dummy_clf_scores(X=X, y=y_true)
    # dict_scores['ga'] = scores(y_true=y_true, y_pred=y_pred)

    # df_scores = pd.DataFrame(dict_scores).T
    # df_scores.sort_values(by='macro_f1_score', ascending=False, inplace=True)
    # print(df_scores)
