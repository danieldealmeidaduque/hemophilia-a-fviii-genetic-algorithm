from auxiliar import math_func


class Gene:
    """Gene is one dataframe grouped by same mutation"""

    N_MATH = 4  # dist * rsa

    def __init__(self, df_key, df_value):
        self.wild_aa, self.new_aa = df_key
        self.df = df_value
        self.fitness = 0

    def __str__(self, show_df=False):
        print(f"\tGENE = {self.wild_aa} -> {self.new_aa}", end=" | ")
        print(f"{len(self.df)} mutations | fit = {self.fitness_mean}", end="")
        # print(f'\n\n{self.df}')
        return ""

    def _getDataFrame(self):
        return self.df

    def _getFitness(self):
        return self.fitness_mean

    def _setFitness(self):
        self.fitness = self.df["fitness"].mean()

    def getYTrue(self):
        return self.df["severity"].values

    def getYPred(self):
        return self.df["fitness_discretized"].values

    def calculateFitness(self, s):
        def fitness(x):
            fitness = math_func[self.N_MATH](x=x, s=s)
            fitness = fitness
            return fitness

        self.df["fitness"] = self.df["rsa"].apply(fitness)

    def normalizeFitness(self):
        fitness = self.df["fitness"].values.copy()
        min, max = fitness.min(), fitness.max()
        if max - min != 0:
            fitness_normalized = [(v - min) / (max - min) for v in fitness]
        else:
            fitness_normalized = (
                -1
            )  # discretize error: jogar fora >> ver outras soluções

        self.df["fitness_normalized"] = fitness_normalized

    def discretizeFitness(self, lb=0.44, ub=0.55):
        def discretize(value):
            if value >= 0 and value < lb:
                return "Mild"  # 0
            elif value >= lb and value < ub:
                return "Moderate"  # 1
            elif value >= ub and value <= 1:
                return "Severe"  # 2
            else:
                return "Moderate"  # discretize error: jogar fora >> ver outras soluções

        self.df["fitness_discretized"] = self.df["fitness_normalized"].apply(discretize)

    def genePredict(self, s):
        self.calculateFitness(s=s)
        self.normalizeFitness()
        self.discretizeFitness(lb=0.33, ub=0.66)
        self._setFitness()
