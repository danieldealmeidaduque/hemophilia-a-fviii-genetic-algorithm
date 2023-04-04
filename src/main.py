from os.path import abspath, dirname, join

import pandas as pd

from ga import GA

if __name__ == "__main__":
    """Get initial data and execute genetic algorithm"""

    # input file paths
    input_dir = abspath(join(dirname(__file__), "..", "datasets"))
    input_file = "champ-mutation-list-q4-clean-enhanced.xlsx"
    input_path = join(input_dir, input_file)

    # input dataframe with all needed information for GA
    df = pd.read_excel("champ-mutation-list-q4-clean-enhanced.xlsx", index_col=0)

    rsa = df["Relative Surface Area"]
    dist = df["Distance Wild and New"]  # substituir pela "solution" do pygad?
    y_true = df['Reported Severity']

    ga = GA(rsa, dist, y_true)  # executar esse pipeline para cada funcao matematica
    ga.calculate_fitness()
    ga.normalize_fitness()
    ga.discretize_fitness()
    ga.solution_fitness()
    ga.plot_confusion_matrix()
