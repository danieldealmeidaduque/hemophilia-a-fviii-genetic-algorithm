from os.path import abspath, dirname, join

import pandas as pd

from chromossome import Chromossome
from gene import Gene

if __name__ == "__main__":
    """Get initial data and execute genetic algorithm"""

    # input file paths
    input_dir = abspath(join(dirname(__file__), "..", "datasets"))
    input_file = "champ-mutation-list-q4-clean-enhanced.xlsx"
    input_path = join(input_dir, input_file)

    # output file paths
    output_dir = abspath(join(dirname(__file__), "..", "workdir"))
    output_ga_file = "ga.pdf"
    output_ga_path = join(output_dir, output_ga_file)

    # input dataframe with all needed informations for GA
    df = pd.read_excel(input_path, index_col=0)
    # print(df)

    # GA gene
    # ga_gene = Gene(df.iloc[0])
    # print(ga_gene)
    # ga_gene.fitness_calculate()
    # print(ga_gene)
    # ga_gene.fitness_discretize()
    # print(ga_gene)

    # GA chromossome
    ga_chromossome = Chromossome(df)
    print(ga_chromossome)
    ga_chromossome.fitness_calculate()
    print(f"Fitness calculation for chromosome")
    print(ga_chromossome)
    ga_chromossome.fitness_normalize()
    print(f"Fitness normalized for chromosome")
    print(ga_chromossome)
    ga_chromossome.fitness_discretize()
    print(f"Fitness discretized for chromosome")
    print(ga_chromossome)
