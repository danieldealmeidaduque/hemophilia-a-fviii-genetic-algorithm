from os.path import abspath, dirname, join
from time import process_time

import pandas as pd
import pygad
from matplotlib import pyplot as plt

from auxiliar import (
    create_confusion_matrix,
    exception_handler,
    finished_time,
    format,
    plot_confusion_matrix,
)
from chromossome import Chromossome, Gene

if __name__ == "__main__":
    """Get initial data and execute genetic algorithm"""

    input_dir = abspath(join(dirname(__file__), "..", "datasets"))
    input_file = "champ-mutation-list-q4-clean-enhanced.xlsx"
    input_path = join(input_dir, input_file)

    df = pd.read_excel(input_path, index_col=0)
    print(df)

    exit(0)

    output_dir = abspath(join(dirname(__file__), "..", "workdir"))
    output_ga_file = f"ga_{NUM_GENERATIONS}_{NUM_PARENTS_MATING}_{SOL_PER_POP}_{PARENT_SELECTION_TYPE}_{KEEP_PARENTS}_{CROSSOVER_TYPE}_{CROSSOVER_PROBABILITY}_{MUTATION_TYPE}_{MUTATION_PERCENT_GENES}.pdf"
    output_ga_path = join(output_dir, output_ga_file)

    # ---- GENETIC ALGORITHM ----
    print()

    df_grouped = df.groupby(["wild_aa", "new_aa"])  # is the chromossome
    genes = [Gene(k, v) for k, v in df_grouped]  # all genes
    chromossome = Chromossome(genes)  # chromossome class

    solution, solution_fitness, solution_idx = ga(chromossome, output_ga_path)
    print(solution_fitness)
    print(solution_idx)
    print(solution)

    """pygad fitness function to give as a parameter"""
    chromossome.chromossomePredict(solution)
    chromossome._setConfusionMatrix()
    chromossome._setFitness()
    chromossome._setSolution(solution)

    cm = chromossome._getConfusionMatrix()
    output_cm_path = join(output_dir, "cm")
    plot_confusion_matrix(cm, output_path=None)
