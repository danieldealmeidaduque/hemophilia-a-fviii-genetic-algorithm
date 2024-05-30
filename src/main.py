import csv
from os.path import abspath, dirname, join
import pandas as pd
from auxiliar import math_funcs
from ga import GA


def load_data(file_path):
    """Load data from the given Excel file path and return necessary columns."""
    try:
        df = pd.read_excel(file_path, index_col=0)
    except Exception as e:
        raise IOError(f"Error loading Excel file: {e}")
    rsa = df["Relative Surface Area"].values
    sev_true = df["Reported Severity"].values
    return rsa, sev_true

def execute_ga(rsa, sev_true):
    """Execute the genetic algorithm for each function in math_funcs."""
    ga = GA(rsa, sev_true)
    solutions = []
    for i in range(len(math_funcs)):
        try:
            solution, solution_fitness, solution_idx = ga.run_pygad(i)
        except Exception as e:
            print(f"Error running GA for function {i}: {e}")
            continue
        print(f"Best solution for function {i}: {solution}")
        print(f"Best solution fitness: {solution_fitness}")
        print(f"Solution index: {solution_idx}")
        solutions.append((i, solution, solution_fitness, solution_idx))
        # Uncomment the next line if you want to plot the confusion matrix
        # ga.plot_confusion_matrix()
    return solutions

def save_solutions(solutions, output_file):
    """Save the solutions to a CSV file."""
    try:
        with open(output_file, "w", encoding="utf-8", newline='') as file:
            writer = csv.writer(file, delimiter="\t", skipinitialspace=True)
            writer.writerow(["Function Index", "Solution", "Fitness", "Solution Index"])  # Header row
            for solution in solutions:
                writer.writerow(solution)
    except Exception as e:
        raise IOError(f"Error writing solutions to CSV file: {e}")

def main():
    """Main function to load data, execute GA, and save solutions."""
    input_dir = abspath(join(dirname(__file__), "..", "datasets"))
    input_file = "champ-mutation-list-q4-clean-enhanced.xlsx"
    input_path = join(input_dir, input_file)

    rsa, sev_true = load_data(input_path)
    solutions = execute_ga(rsa, sev_true)
    output_file = "results/solutions.csv"
    save_solutions(solutions, output_file)

if __name__ == "__main__":
    main()
