from auxiliar import func2str, math_funcs


class Gene:
    """Gene is a filtered dataframe row object"""

    def __init__(self, row, n_math_func=4):
        self.wild = str(row["HGVS Wild Amino Acid"])
        self.new = str(row["HGVS New Amino Acid"])
        self.pos = int(row["HGVS Position"])
        self.rsa = float(row["Relative Surface Area"])
        self.dist = float(row["Distance Wild and New"])
        self.sev = str(row["Reported Severity"])
        self.n_math_func = n_math_func
        self.fitness = None

    def __str__(self):
        print(func2str(math_funcs[self.n_math_func]), end=" | ")
        return f"Gene: {self.wild} -> {self.new} | Fitness = {self.fitness}"

    def fitness_calculate(self):
        n = self.n_math_func
        d = self.dist
        r = self.rsa
        f = math_funcs[n](d, r)
        self.fitness = f

    def fitness_discretize(self, lower_bound=0.33, upper_bound=0.66):
        f = self.fitness
        if f > 0 and f <= lower_bound:
            self.fitness = "Mild"
        elif f > lower_bound and f <= upper_bound:
            self.fitness = "Moderate"
        elif f > upper_bound and f <= 1:
            self.fitness = "Severe"
        else:
            self.fitness = None
