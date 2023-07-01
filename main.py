import argparse
import time
import logging
from multiprocessing import Pool

from evo.evolution.algorithms import StochasticSolver
from evo.evolution.objectives import ObjectiveDict
from evo.listeners.listener import FileListener


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--np", type=int, default=1, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--n_params", type=int, default=2, help="solution size")
    parser.add_argument("--evals", type=int, default=2500, help="fitness evaluations")
    return parser.parse_args()


def parallel_solve(solver, iterations, config, listener):
    num_workers = config.np
    if solver.pop_size % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    for j in range(iterations):
        solutions = solver.ask()
        print(len(solutions))
        with Pool(num_workers) as pool:
            results = pool.map(parallel_wrapper, [(config, solutions[i], i) for i in range(solver.pop_size)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        print(len(solver.pop))
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": j * solver.pop_size, "best.fitness": result[1],
                           "best.solution": result[0]})
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
    return best_result, best_fitness


def parallel_wrapper(arg):
    c, solution, i = arg
    fitness = evaluate(config=c, solution=solution)
    return i, -fitness


def evaluate(config, solution):
    return 0.0  # fitness


if __name__ == "__main__":
    args = parse_args()
    file_name = ".".join([args.solver, str(args.s)])
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=5.0)
    listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
                                                         "best.solution"])
    solver = StochasticSolver.create_solver(name=args.solver,
                                            seed=args.s,
                                            num_params=args.n_params,
                                            pop_size=100,
                                            genotype_factory="uniform_float",
                                            solution_mapper="direct",
                                            objectives_dict=objectives_dict,
                                            offspring_size=100,
                                            remap=False,
                                            genetic_operators={"gaussian_mut": 1.0},
                                            genotype_filter=None,
                                            tournament_size=5,
                                            mu=0.0,
                                            sigma=0.35,
                                            n=args.n_params,
                                            range=(-1, 1),
                                            upper=2.0,
                                            lower=-1.0)
    best = parallel_solve(solver=solver, iterations=args.evals // solver.pop_size, config=args, listener=listener)
