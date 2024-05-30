import inspect
import re
import sys
from math import sqrt
from time import process_time

# ------------------------ Mathematical Functions -------------------------------- #


math_funcs = {
    0: lambda dist, rsa: abs(dist - rsa),
    1: lambda dist, rsa: dist + rsa,
    2: lambda dist, rsa: pow(dist, 2) + rsa,
    3: lambda dist, rsa: pow(dist, 3) + rsa,
    4: lambda dist, rsa: dist * rsa,
    5: lambda dist, rsa: pow(dist, 2) * rsa,
    6: lambda dist, rsa: pow(dist, 3) * rsa,
    7: lambda dist, rsa: pow(dist, rsa),
    8: lambda dist, rsa: abs(pow(dist, rsa) - abs(dist - rsa)),
    9: lambda dist, rsa: abs(pow(dist, 2) - abs(dist - rsa)),
    10: lambda dist, rsa: abs(pow(dist, 3) - abs(dist - rsa)),
    11: lambda dist, rsa: abs(pow(dist, abs(dist - rsa)) - abs(dist - rsa)),
    12: lambda dist, rsa: sqrt(abs(dist - rsa)),
    13: lambda dist, rsa: sqrt(abs(dist + rsa)),
    14: lambda dist, rsa: sqrt(abs(pow(dist, 2) + rsa)),
    15: lambda dist, rsa: sqrt(abs(pow(dist, 3) + rsa)),
    16: lambda dist, rsa: sqrt(abs(dist * rsa)),
    17: lambda dist, rsa: sqrt(abs(pow(dist, 2) * rsa)),
    18: lambda dist, rsa: sqrt(abs(pow(dist, 3) * rsa)),
    19: lambda dist, rsa: sqrt(abs(pow(dist, rsa))),
    20: lambda dist, rsa: sqrt(abs(pow(dist, rsa) - abs(dist - rsa))),
    21: lambda dist, rsa: sqrt(abs(pow(dist, 2) - abs(dist - rsa))),
    22: lambda dist, rsa: sqrt(abs(pow(dist, 3) - abs(dist - rsa))),
    23: lambda dist, rsa: sqrt(abs(pow(dist, abs(dist - rsa)) - abs(dist - rsa))),
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
def highlight(s):
    """print upper and lower highlighted text"""
    print("\n\t\t\t ----------- " + s.upper() + " ------------\n")


@exception_handler
def finished_time(start_time, finish_msg):
    """print finished time formatted"""
    f_time = process_time()
    e_time = round((f_time - start_time), 2)
    print(f"\n\t{finish_msg} - Finished  in {e_time} s.\n")


@exception_handler
def func2str(func):
    """convert one line of math_func to a string"""
    line = inspect.getsourcelines(func)[0][0]
    return re.search("rsa: .*", line).group()[5:-1].strip()


