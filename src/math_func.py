from math import sqrt

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
