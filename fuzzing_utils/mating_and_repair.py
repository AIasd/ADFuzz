from pymoo.model.mating import Mating
from pymoo.model.repair import Repair
import numpy as np

class MyMatingVectorized(Mating):
    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 use_unique_bugs,
                 emcmc,
                 mating_max_iterations,
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.use_unique_bugs = use_unique_bugs
        self.mating_max_iterations = mating_max_iterations
        self.emcmc = emcmc

    def do(self, problem, pop, n_offsprings, **kwargs):

        if self.mating_max_iterations >= 5:
            mating_max_iterations = self.mating_max_iterations // 5
            n_offsprings_sampling = n_offsprings * 5
        else:
            mating_max_iterations = self.mating_max_iterations
            n_offsprings_sampling = n_offsprings

        # the population object to be used
        off = pop.new()
        parents = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:
            n_infills += 1
            print('n_infills / mating_max_iterations', n_infills, '/', mating_max_iterations, 'len(off)', len(off))
            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills >= mating_max_iterations:
                break

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off, _parents = self._do(problem, pop, n_offsprings_sampling, **kwargs)

            # repair the individuals if necessary - disabled if repair is NoRepair
            _off_first = self.repair.do(problem, _off, **kwargs)

            # Vectorized
            _off_X = np.array([x.X for x in _off_first])
            remaining_inds = if_violate_constraints_vectorized(_off_X, problem.customized_constraints, problem.labels, problem.ego_start_position, verbose=False)
            _off_X = _off_X[remaining_inds]

            _off = _off_first[remaining_inds]
            _parents = _parents[remaining_inds]

            # Vectorized
            if self.use_unique_bugs:
                if len(_off) == 0:
                    continue
                elif len(off) > 0 and len(problem.interested_unique_bugs) > 0:
                    prev_X = np.concatenate([problem.interested_unique_bugs, np.array([x.X for x in off])])
                elif len(off) > 0:
                    prev_X = np.array([x.X for x in off])
                else:
                    prev_X = problem.interested_unique_bugs

                print('\n', 'MyMating len(prev_X)', len(prev_X), '\n')
                remaining_inds = is_distinct_vectorized(_off_X, prev_X, problem.mask, problem.xl, problem.xu, problem.p, problem.c, problem.th, verbose=False)

                if len(remaining_inds) == 0:
                    continue

                _off = _off[remaining_inds]
                _parents = _parents[remaining_inds]
                assert len(_parents)==len(_off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]
                _parents = _parents[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            parents = Population.merge(parents, _parents)

        # assert len(parents)==len(off)
        print('Mating finds', len(off), 'offsprings after doing', n_infills, '/', mating_max_iterations, 'mating iterations')
        return off, parents

    # only to get parents
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:
            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)
            parents_obj = pop[parents].reshape([-1, 1]).squeeze()
        else:
            parents_obj = parents

        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)
        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off, parents_obj

class ClipRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        for i in range(len(pop)):
            pop[i].X = np.clip(pop[i].X, np.array(problem.xl), np.array(problem.xu))
        return pop
