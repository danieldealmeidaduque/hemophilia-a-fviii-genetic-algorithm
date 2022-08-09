import re
import sys
import inspect
import numpy as np
import seaborn as sns
from time import process_time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score

sev2int = {
    'Mild': 0,
    'Moderate': 1,
    'Severe': 2
}

int2sev = {
    0: 'Mild',
    1: 'Moderate',
    2: 'Severe'
}

# ------------------------ Auxiliar Functions ------------------------------------ #


def format(value):
    return round(value, 3)


def exception_handler(func):
    '''cleaner way to handle exceptions and exit the program'''
    def inner_function(*args, **kwargs):
        try:
            retry = func(*args, **kwargs)
            return retry
        except:
            print(f'\n\t !!!!!!! ERROR IN FUNCTION: {func.__name__}\n')
            sys.exit(1)
    return inner_function


@ exception_handler
def finished_time(start_time, finish_msg):
    '''print finished time formatted'''
    f_time = process_time()
    e_time = format((f_time - start_time))
    msg = f'\n\t{finish_msg} - Finished  in {e_time} s.\n'
    print(msg)


@ exception_handler
def highlight(s):
    '''print upper and lower highlighted text'''
    print('\n\t\t\t ----------- ' + s.upper() + ' ------------\n')


@ exception_handler
def math_func2string(func):
    '''convert one line of math_func to a string'''
    line = inspect.getsourcelines(func)[0][0]
    str_func = re.search('s:.*', line).group()[3:-1].strip()

    return str_func


def strip_string(s):
    try:
        s = s.strip()
    finally:
        return s

# ------------------------ Confusion Matrix Functions ---------------------------- #


@ exception_handler
def create_cf_matrix(y_true, y_pred):
    '''create the confusion matrix'''
    return confusion_matrix(y_true, y_pred)


@ exception_handler
def sum_diagonal_cf_matrix(cf_matrix):
    '''sum of principal diagonal of the confusion matrix'''
    n = len(cf_matrix)
    m = len(cf_matrix[0])

    sum_corrects = 0
    for i in range(n):
        for j in range(m):
            if i == j:
                correct = cf_matrix[i][j].sum()
                sum_corrects += correct

    return sum_corrects


@ exception_handler
def plot_cf_matrix(cf_matrix, save=False):
    '''plot confusion matrix as heatmap'''
    fig, ax = plt.subplots(figsize=(8, 8))

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix')
    ax.set_xlabel('Predicted Hemophilia Severity')
    ax.set_ylabel('Actual Hemophilia Severity')

    # labels - list must be in alphabetical order
    ax.xaxis.set_ticklabels(['Mild', 'Mod', 'Sev'])
    ax.yaxis.set_ticklabels(['Mild', 'Mod', 'Sev'])

    if save:
        plt.savefig('cf_matrix.png')
    else:
        plt.show()

# ------------------------ Normalization and Discretization ---------------------- #


def min_max_normalization(V):
    '''normalize vector using min-max normalization'''
    min_v = min(V)
    max_v = max(V)
    if max_v - min_v != 0:
        for i, v in enumerate(V):
            V[i] = (v - min_v) / (max_v - min_v)
    return V


@ exception_handler
def discretize_to_severity(vector, lb=0.33, ub=0.66):  # 0.44 and 0.55?
    '''function to discretize a 1D array to severity (0, 1, 2)'''
    def cut(v):
        '''function to cut using lower and upper bounds'''
        if v >= 0 and v <= lb:
            return 0  # 'Mild'
        elif v > lb and v <= ub:
            return 1  # 'Moderate'
        else:
            return 2  # 'Severe'

    S = [cut(v) for v in vector]

    return S

# ----------------------------- Prediction Scores -------------------------------- #


@ exception_handler
def scores(y_true, y_pred):
    '''function to calculate several scores based on y_true and y_pred'''
    avg = 'macro'

    acc = format(accuracy_score(y_true, y_pred))
    b_acc = format(balanced_accuracy_score(y_true, y_pred))

    p = format(precision_score(y_true, y_pred, average=avg, zero_division=1))
    f1 = format(f1_score(y_true, y_pred, average=avg, zero_division=1))

    dict_scores = {
        'accuracy_score': acc,  # = micro_f1_score, micro_recall_score and micro_precision_score
        'balanced_accuracy_score': b_acc,  # = macro_recall_score
        'macro_precision_score': p,
        'macro_f1_score': f1,
    }

    return dict_scores

# -- GA --


@ exception_handler
def get_initial_pop(df, size):
    '''get initial population for the ga'''
    initial_pop = [np.array(df.dist_aa.values)] * size

    print(f'\nsize: ind, pop = {len(initial_pop[0])}, {len(initial_pop)}\n')
    return initial_pop


@ exception_handler
def get_X(df):
    '''get X for the ga'''
    X = df.dist_aa.values

    print(f'\nX = {X[:5]} len:{len(X)}')
    return X


@ exception_handler
def get_Y_true(df):
    '''get Y true for the ga'''
    y_true = np.array([sev2int[i] for i in list(df.severity.values)])

    print(f'y = {y_true[:5]} len:{len(y_true)}\n')
    return y_true
