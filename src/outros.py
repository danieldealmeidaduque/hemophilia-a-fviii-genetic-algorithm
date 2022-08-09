import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from auxiliar import scores, sum_diagonal_cf_matrix, plot_cf_matrix
from auxiliar import exception_handler, highlight
from math_func import math_func


# ------------------------ Fitness testing functions ----------------------------- #


@ exception_handler
def insert_fitness_value(df, function):
    '''apply generic function passed as parameter to calculate fitness'''
    def calculate_fitness_value(row):
        dist = row.dist_aa
        rsa = row.rsa

        fitness = function(dist, rsa)

        return fitness

    df['fitness'] = df.apply(calculate_fitness_value, axis=1)


@ exception_handler
def insert_severity_based_on_fitness_value(df, lower_bound_cutoff=33, upper_bound_cutoff=66):
    '''insert predicted severity in dataframe based on fitness value'''
    # sort dataframe by fitness value
    df.sort_values(by='fitness', inplace=True)

    # get bounds based on cutoff and size of the dataframe
    lower_bound = int(np.floor(df.shape[0] * (lower_bound_cutoff/100)))
    upper_bound = int(np.ceil(df.shape[0] * (upper_bound_cutoff/100)))

    # quantity of mil, mod, sev
    qnt_mil = ['Mild'] * lower_bound
    qnt_mod = ['Moderate'] * (upper_bound - lower_bound)
    qnt_sev = ['Severe'] * (df.shape[0] - upper_bound)

    # since is ordered by fitness just need to append all severities
    df['predicted_severity'] = qnt_mil + qnt_mod + qnt_sev


@exception_handler
def automatic_best_fitness(df):
    '''function to automated find best population fitness'''
    def automatic_pop_fitness(df, lower_bound, upper_bound):
        '''function to automated calculate population fitness'''
        # add column with predicted severity
        insert_severity_based_on_fitness_value(df, lower_bound, upper_bound)
        # create confusion matrix
        y_true = df.severity
        y_pred = df.predicted_severity
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        # calculate population fitness
        pop_fitness = sum_diagonal_cf_matrix(cf_matrix)

        return pop_fitness

    # use the function above to calculate all population fitness
    list_pop_fitness = []
    initial_lower_bound = 1
    for lower_bound in range(initial_lower_bound, 50):
        # lb - up = 1% - 99% -> 2% - 98% .. -> 49% - 51%
        upper_bound = 100 - lower_bound

        # calculate fitness using this bounds
        pop_fitness = automatic_pop_fitness(df, lower_bound, upper_bound)

        # add the fitness and their bound to a list
        element_to_add = {
            'pop_fitness': pop_fitness,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        list_pop_fitness.append(element_to_add)

    # df with pop fitness, lower bound and upper bound
    df_fit = pd.DataFrame(list_pop_fitness)

    # df with only the max fitness value, lower bound and upper bound
    max_fit = df_fit.pop_fitness == max(df_fit.pop_fitness)
    df_max_fit = df_fit[max_fit]

    return df_max_fit


@ exception_handler
def testing_fitness_functions_without_ga(df):
    '''testing some fitness functions before genetic algorithm'''
    # copy to keep original intact
    df_test = df.copy()

    # calculate fitness value for each mutation
    insert_fitness_value(df_test, math_func[10])
    highlight('Dataframe with fitness values')
    print(df_test)

    # find best population fitness_value, lower bound and upper bound
    df_best_fit = automatic_best_fitness(df_test)
    lb = df_best_fit.lower_bound
    ub = df_best_fit.upper_bound
    print(df_best_fit)

    # using best bounds to write fitness as severity
    insert_severity_based_on_fitness_value(df_test, lb, ub)
    highlight('Dataframe with fitness as severity')
    print(df_test)

    # create confusion matrix
    y_true = df_test.severity
    y_pred = df_test.predicted_severity
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    highlight('Plotting confusion matrix')
    plot_cf_matrix(cf_matrix)

    # using confusion matrix to calculate population fitness
    pop_fitness_value = sum_diagonal_cf_matrix(cf_matrix)
    print(pop_fitness_value)

# ------------------------ Plot functions ---------------------------------------- #


@ exception_handler
def plots_to_get_insights(df):
    # # countplot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # fig.suptitle('Countplot')

    # sns.countplot(ax=ax1, data=df, x='severity')
    # ax1.set_title('True Severity')

    # sns.countplot(ax=ax2, data=df, x='severity_fitness')
    # ax2.set_title('Predict Severity')
    # plt.show()

    # boxplot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # fig.suptitle('Boxplot - Severity')

    # sns.boxplot(ax=ax1, data=df, x='severity', y='dist_aa')
    # ax1.set_title('Severity x dist')

    # sns.boxplot(ax=ax2, data=df, x='severity', y='rsa')
    # ax2.set_title('Severity x rsa')
    # plt.show()

    sns.boxplot(data=df, x='severity', y='dist_aa')
    plt.title('Severity x dist')
    plt.show()

    sns.boxplot(data=df, x='severity', y='rsa')
    plt.title('Severity x rsa')
    plt.show()

    # # scatterplot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # fig.suptitle('Scatterplot - Distance x RSA')

    # sns.scatterplot(ax=ax1, data=df, x='dist_aa', y='rsa', hue='severity')
    # ax1.set_title('True Severity')

    # sns.scatterplot(ax=ax2, data=df, x='dist_aa',
    #                 y='rsa', hue='severity_fitness')
    # ax2.set_title('Predict Severity')
    # plt.show()

    # # order to histogram plot
    # df.sort_values(by='fitness', ascending=False, inplace=True)

    # # histplot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # fig.suptitle('Histplot')

    # sns.histplot(ax=ax1, data=df, x='fitness', hue='severity')
    # ax1.set_title('True Severity')

    # sns.histplot(ax=ax2, data=df, x='fitness', hue='severity_fitness')
    # ax2.set_title('Predict Severity')
    # plt.show()


@ exception_handler
def dummy_clf_scores(X, y):
    '''function to generate scores using dummy classifiers'''
    strategies = ['most_frequent', 'prior', 'stratified', 'uniform']
    dict_scores = {}

    for s in strategies:
        # make the prediction
        dummy_clf = DummyClassifier(strategy=s, random_state=42)
        dummy_clf.fit(X, y)
        y_pred = dummy_clf.predict(X)

        # score of the prediction
        dummy_score = scores(y_true=y, y_pred=y_pred)
        dict_scores[s] = dummy_score

    return dict_scores

# # ------------------------ Input Diretory and Files ------------------------------ #

# # diretory
# input_dir = abspath(join(dirname(__file__), '..', 'datasets'))

# # input file - point mutations (pm)
# pm_file = 'FVIII_point_mutations_v1.csv'
# pm_path = join(input_dir, pm_file)

# # input file - amino acids distance matrix (dm)
# dm_file = 'Supplementary_Table_npj_paper.xlsx'
# dm_path = join(input_dir, dm_file)

# # input file - relative surface area (rsa)
# rsa_file = 'Relative_Surf_Area_2R7E_v2.csv'
# rsa_path = join(input_dir, rsa_file)

# # dataframe with all informations
# df = get_initial_df(pm_path, dm_path, rsa_path)
# print(f'\n{df}')

# # dataframe filtered to use in ga
# df_unique_mut = filter_unique_mutations(df)
# print(f'df_filtered:\n\n{df_unique_mut}')

# # ------------------ Recreating Distance Matrix with GA solution ---

# # ga solution to distance matrix
# df_unique_mut['dist_aa'] = solution
#  # index and column names
#  aa_list = np.unique(df_unique_mut['wild_aa'].values.tolist())
#   # empty distance matrix labeled
#   dm = pd.DataFrame(index=aa_list, columns=aa_list)

#    # function to insert an amino acid distance in distance matrix
#    def insert_aa_in_dm(row):
#         dm.at[row.wild_aa, row.new_aa] = round(row.dist_aa, 2)

#     df_unique_mut.apply(insert_aa_in_dm, axis=1)

#     print(df.describe())
#     print(df_unique_mut.describe())
#     print(dm)

#     # # ------------------ Other Classifiers and Scores ------------------
#     # dict_scores = {}
#     # dict_scores = dummy_clf_scores(X=X, y=y_true)
#     # dict_scores['ga'] = scores(y_true=y_true, y_pred=y_pred)

#     # df_scores = pd.DataFrame(dict_scores).T
#     # df_scores.sort_values(by='macro_f1_score', ascending=False, inplace=True)
#     # print(df_scores)
