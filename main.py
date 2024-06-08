from mpi4py import MPI
from calculator import SequentialTrapezoidStrategy, rosenbrock_fun, ParallelTrapezoidStrategy, \
    SequentialMonteCarloStrategy, ParallelMonteCarloStrategy

if __name__ == '__main__':
    result = ParallelMonteCarloStrategy(-200, 200, -100, 300, 30000000).calculate(rosenbrock_fun)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(result)
