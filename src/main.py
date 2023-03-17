from os.path import abspath, dirname, join

import pandas as pd
from ga import GA
from sklearn import preprocessing

from auxiliar import math_funcs
from chromosome import Chromosome
from gene import Gene


def fitness_calculate(row, n=4):
    d = row["Distance Wild and New"]
    r = row["Relative Surface Area"]
    f = math_funcs[n](d, r)
    return f


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
    print(df)

    for i in range(0, len(math_funcs)):
        df[f"Mutation Fitness - Math Func[{i}]"] = df.apply(
            lambda row: fitness_calculate(row, i), axis=1
        )
    print(df)

    # GA gene
    # ga_gene = Gene(df.iloc[0])
    # print(ga_gene)
    # ga_gene.fitness_calculate()
    # print(ga_gene)
    # ga_gene.fitness_discretize()
    # print(ga_gene)

    # GA chromosome
    # ga_chromosome = Chromosome(df)
    # print(ga_chromosome)
    # ga_chromosome.fitness_calculate()
    # print(f"Fitness calculation for chromosome")
    # print(ga_chromosome)
    # ga_chromosome.fitness_normalize()
    # print(f"Fitness normalized for chromosome")
    # print(ga_chromosome)
    # ga_chromosome.fitness_discretize()
    # print(f"Fitness discretized for chromosome")
    # print(ga_chromosome)
    # ga_chromosome.fitness_solution()
    # print(f"Fitness solution for chromosome")
    # print(ga_chromosome)
