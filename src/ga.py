from os.path import abspath, dirname, join
from time import process_time

import pygad

from auxiliar import exception_handler, finished_time, format, plot_confusion_matrix
from chromossome import Chromossome
from gene import Gene


@exception_handler
class GA:
    """function to execute the genetic algorithm"""

    # ---- GA CONSTANTS ----
    # 50 generations + 50 sol per pop = 5 min de execucao
    # 50 generations + 100 sol per pop = 15 min de execucao

    # number of generations
    NUM_GENERATIONS = 50  # 50
    # number of solutions to be selected as parents
    NUM_PARENTS_MATING = 5  # 5
    # number of solutions (chromossomes)
    SOL_PER_POP = 50  # 50
    # sss | rws | rank | tournament
    PARENT_SELECTION_TYPE = "rank"  # rank
    # number of parents to keep in the current population
    KEEP_PARENTS = 1  # 1
    # single_point | two_points | uniform | scattered | a custom crossover function
    CROSSOVER_TYPE = "uniform"  # uniform
    # probability of crossover
    CROSSOVER_PROBABILITY = 0.9  # 0.9
    # random | swap | inversion | scramble | adaptive | a custom mutation function
    MUTATION_TYPE = "random"  # random
    # percentage of genes to mutate
    MUTATION_PERCENT_GENES = 10  # 10
    # gene limit values
    # GENE_SPACE = {'low': 0, 'high': 10}
    # -1 e 1 roda 10 vezes e olha os parametros >> aumenta o intervalo com base nos resultados

    def __init__(self, df, output_dir):
        print("Initiating GA...")
        self.output_dir = output_dir
        ga_start_time = process_time()
        df_grouped = df.groupby(["wild_aa", "new_aa"])  # is the chromossome
        self.genes = [Gene(k, v) for k, v in df_grouped]  # all genes
        self.chromossome = Chromossome(self.genes)  # chromossome class

    def __str__(self):
        cm = self.chromossome._getConfusionMatrix()
        output_cm_path = join(self.output_dir, "cm")
        plot_confusion_matrix(cm, output_path=None)
        return str(output_cm_path)

    def fitness(self, solution, solution_idx):
        """pygad fitness function to give as a parameter"""
        self.chromossome.chromossomePredict(solution)
        self.chromossome._setConfusionMatrix()
        self.chromossome._setFitness()
        self.chromossome._setSolution(solution)

        self.solution_fitness = self.chromossome._getFitness()
        self.solution = solution
        self.solution_idx = solution_idx
        return self.solution_fitness

    def prepare(self):
        """function to prepare ga parameters"""
        print("Preparing GA...")
        num_genes = len(self.chromossome._getGenes())
        fitness_function = self.fitness()

        def on_generation(ga_instance):
            fit = self.chromossome._getFitness()
            # s = self.chromossome._getSolution()
            print(f"\n GENERATION - Fitness: {format(fit)}")
            print(self.solution_fitness)
            print(self.solution_idx)
            print(self.solution)

        ga_instance = pygad.GA(
            num_generations=self.NUM_GENERATIONS,
            num_parents_mating=self.NUM_PARENTS_MATING,
            fitness_func=fitness_function,
            sol_per_pop=self.SOL_PER_POP,
            num_genes=num_genes,
            parent_selection_type=self.PARENT_SELECTION_TYPE,
            keep_parents=self.KEEP_PARENTS,
            crossover_type=self.CROSSOVER_TYPE,
            crossover_probability=self.CROSSOVER_PROBABILITY,
            on_generation=on_generation,
            mutation_type=self.MUTATION_TYPE,
            mutation_percent_genes=self.MUTATION_PERCENT_GENES,
            #    gene_space=GENE_SPACE # fitness ficou fixo..
        )

        return ga_instance

    def execute(self):
        # prepare ga instance
        ga_instance = self.prepare()
        # run ga instance
        ga_instance.run()
        # get output parameters
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        # plot fitness function per generation
        ga_instance.plot_fitness(save_dir=self.output_path)

        finished_time(self.ga_start_time, "GA ALGORITHM")
