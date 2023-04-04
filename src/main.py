import csv
from os.path import abspath, dirname, join

import pandas as pd

from auxiliar import math_funcs
from ga import GA

if __name__ == "__main__":
    """Get initial data and execute genetic algorithm"""

    # input file paths
    input_dir = abspath(join(dirname(__file__), "..", "datasets"))
    input_file = "champ-mutation-list-q4-clean-enhanced.xlsx"
    input_path = join(input_dir, input_file)

    # input dataframe with all needed information for GA
    df = pd.read_excel(input_path, index_col=0)

    # get only needed information for GA
    rsa = df["Relative Surface Area"].values
    sev_true = df["Reported Severity"].values

    # executing GA with given information
    solutions = []
    ga = GA(rsa, sev_true)
    for i in range(len(math_funcs)):
        solution, solution_fitness, solution_idx = ga.pygad(i)
        solutions.append((i, solution_idx, solution_fitness))
        # break

    with open("solutions.csv", "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t", skipinitialspace=True)
        for solution in solutions:
            writer.writerow(solution)
