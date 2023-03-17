from os.path import abspath, dirname, join

import pandas as pd

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
