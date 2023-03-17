import inspect
import re
import sys
from math import sqrt
from time import process_time

from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
)

# ------------------------ Mathematical Functions -------------------------------- #


math_func = {
    0: lambda x, s: abs(x - s),
    1: lambda x, s: x + s,
    2: lambda x, s: pow(x, 2) + s,
    3: lambda x, s: pow(x, 3) + s,
    4: lambda x, s: x * s,
    5: lambda x, s: pow(x, 2) * s,
    6: lambda x, s: pow(x, 3) * s,
    7: lambda x, s: pow(x, s),
    8: lambda x, s: pow(x, pow(s, 2)),
    9: lambda x, s: pow(x, 2) + pow(s, 2),
    10: lambda x, s: pow(x, 3) + pow(s, 2),
    11: lambda x, s: abs(pow(x, s) - abs(x - s)),
    12: lambda x, s: abs(pow(x, 2) - abs(x - s)),
    13: lambda x, s: abs(pow(x, 3) - abs(x - s)),
    14: lambda x, s: abs(pow(x, pow(s, 2)) - abs(x - s)),
    15: lambda x, s: abs(pow(x, abs(x - s)) - abs(x - s)),
    16: lambda x, s: abs(pow(x, pow(abs(x - s), 2)) - abs(x - s)),
    17: lambda x, s: abs(pow(x, pow(abs(x - s), 3)) - abs(x - s)),
    18: lambda x, s: sqrt(abs(x - s)),
    19: lambda x, s: sqrt(abs(x + s)),
    20: lambda x, s: sqrt(abs(pow(x, 2) + s)),
    21: lambda x, s: sqrt(abs(pow(x, 3) + s)),
    22: lambda x, s: sqrt(abs(x * s)),
    23: lambda x, s: sqrt(abs(pow(x, 2) * s)),
    24: lambda x, s: sqrt(abs(pow(x, 3) * s)),
    25: lambda x, s: sqrt(abs(pow(x, s))),
    26: lambda x, s: sqrt(abs(pow(x, pow(s, 2)))),
    27: lambda x, s: sqrt(abs(pow(x, 2) + pow(s, 2))),
    28: lambda x, s: sqrt(abs(pow(x, 3) + pow(s, 2))),
    29: lambda x, s: sqrt(abs(pow(x, s) - abs(x - s))),
    30: lambda x, s: sqrt(abs(pow(x, 2) - abs(x - s))),
    31: lambda x, s: sqrt(abs(pow(x, 3) - abs(x - s))),
    32: lambda x, s: sqrt(abs(pow(x, pow(s, 2)) - abs(x - s))),
    33: lambda x, s: sqrt(abs(pow(x, abs(x - s)) - abs(x - s))),
    34: lambda x, s: sqrt(abs(pow(x, pow(abs(x - s), 2)) - abs(x - s))),
    35: lambda x, s: sqrt(abs(pow(x, pow(abs(x - s), 3)) - abs(x - s))),
}


# ------------------------ Auxiliar Functions ------------------------------------ #


def exception_handler(func):
    """cleaner way to handle exceptions and exit the program"""

    def inner_function(*args, **kwargs):
        try:
            retry = func(*args, **kwargs)
            return retry
        except:
            print(f"\n\t !!!!!!! ERROR IN FUNCTION: {func.__name__}\n")
            sys.exit(1)

    return inner_function


@exception_handler
def format(value):
    return round(value, 3)


@exception_handler
def highlight(s):
    """print upper and lower highlighted text"""
    print("\n\t\t\t ----------- " + s.upper() + " ------------\n")


@exception_handler
def strip_string(s):
    try:
        s = s.strip()
    finally:
        return s


@exception_handler
def finished_time(start_time, finish_msg):
    """print finished time formatted"""
    f_time = process_time()
    e_time = format((f_time - start_time))
    msg = f"\n\t{finish_msg} - Finished  in {e_time} s.\n"
    print(msg)


@exception_handler
def math_func2string(func):
    """convert one line of math_func to a string"""
    line = inspect.getsourcelines(func)[0][0]
    str_func = re.search("s:.*", line).group()[3:-1].strip()

    return str_func


# ----------------------------- Prediction Scores -------------------------------- #


@exception_handler
def scores(y_true, y_pred):
    """function to calculate several scores based on y_true and y_pred"""
    # print(y_true, y_pred)

    avg = "macro"

    acc = format(accuracy_score(y_true, y_pred))
    b_acc = format(balanced_accuracy_score(y_true, y_pred))

    p = format(precision_score(y_true, y_pred, average=avg, zero_division=1))
    f1 = format(f1_score(y_true, y_pred, average=avg, zero_division=1))

    dict_scores = {
        "accuracy_score": acc,  # = micro_f1_score, micro_recall_score and micro_precision_score
        "balanced_accuracy_score": b_acc,  # = macro_recall_score
        "macro_precision_score": p,
        "macro_f1_score": f1,
    }

    return dict_scores


# ----------------------------- Confusion Matrix --------------------------------- #


# @ exception_handler
def create_confusion_matrix(y_true, y_pred, normalize=None, plot=False):
    labels = ["Mild", "Moderate", "Severe"]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if plot:
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot()
        plt.title("Confusion Matrix normalized by row")
        plt.show()

    return cm


# @ exception_handler
def plot_confusion_matrix(cm, output_path=None):
    labels = ["Mild", "Moderate", "Severe"]

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion Matrix normalized by row")
    if output_path == None:
        plt.show()
    else:
        plt.savefig(output_path)
