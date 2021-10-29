from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair


class RandomAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 individual=Individual(),
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # number of offsprings
        self.n_offsprings = pop_size

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = NoRepair()

        self.initialization = Initialization(sampling,
                                             individual=individual,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):
        # create the initial population
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        self.pop, self.off = pop, pop

    def _next(self):
        # sampling again
        self.off = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        self.pop = self.off
