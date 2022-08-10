import re
import sys
import inspect
from time import process_time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score


# ------------------------ Auxiliar Functions ------------------------------------ #


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
def format(value):
    return round(value, 3)


@ exception_handler
def highlight(s):
    '''print upper and lower highlighted text'''
    print('\n\t\t\t ----------- ' + s.upper() + ' ------------\n')


@ exception_handler
def strip_string(s):
    try:
        s = s.strip()
    finally:
        return s


@ exception_handler
def finished_time(start_time, finish_msg):
    '''print finished time formatted'''
    f_time = process_time()
    e_time = format((f_time - start_time))
    msg = f'\n\t{finish_msg} - Finished  in {e_time} s.\n'
    print(msg)


@ exception_handler
def math_func2string(func):
    '''convert one line of math_func to a string'''
    line = inspect.getsourcelines(func)[0][0]
    str_func = re.search('s:.*', line).group()[3:-1].strip()

    return str_func


# ----------------------------- Prediction Scores -------------------------------- #


# @ exception_handler
def scores(y_true, y_pred):
    '''function to calculate several scores based on y_true and y_pred'''
    # print(y_true, y_pred)

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

# ----------------------------- Confusion Matrix --------------------------------- #


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
