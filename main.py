import argparse
import time

from mpi4py import MPI
from calculator import (
    SequentialTrapezoidStrategy,
    rosenbrock_fun,
    ParallelTrapezoidStrategy,
    SequentialMonteCarloStrategy,
    ParallelMonteCarloStrategy,
    ricker_wavelet,
    schwefel_fun,
)


def get_function(name):
    functions = {
        "rosenbrock": rosenbrock_fun,
        "ricker_wavelet": ricker_wavelet,
        "schwefel": schwefel_fun,
    }
    return functions.get(name, rosenbrock_fun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run integration strategies with specified parameters."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "SequentialTrapezoid",
            "ParallelTrapezoid",
            "SequentialMonteCarlo",
            "ParallelMonteCarlo",
        ],
        default="SequentialTrapezoid",
        help="Mode of integration strategy.",
    )
    parser.add_argument(
        "--function",
        type=str,
        choices=["rosenbrock", "ricker_wavelet", "schwefel"],
        default="rosenbrock",
        help="Function to integrate.",
    )
    parser.add_argument(
        "--ax", type=float, default=0, help="Lower bound for x-axis."
    )
    parser.add_argument(
        "--bx", type=float, default=1, help="Upper bound for x-axis."
    )
    parser.add_argument(
        "--ay", type=float, default=0, help="Lower bound for y-axis."
    )
    parser.add_argument(
        "--by", type=float, default=1, help="Upper bound for y-axis."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of intervals or points.",
    )

    args = parser.parse_args()

    function = get_function(args.function)

    if args.mode == "ParallelTrapezoid":
        strategy_class = ParallelTrapezoidStrategy
    elif args.mode == "SequentialMonteCarlo":
        strategy_class = SequentialMonteCarloStrategy
    elif args.mode == "ParallelMonteCarlo":
        strategy_class = ParallelMonteCarloStrategy
    else:
        strategy_class = SequentialTrapezoidStrategy

    if MPI.COMM_WORLD.Get_rank() == 0:
        start_time = time.time()

    result = strategy_class(args.ax, args.bx, args.ay, args.by, args.n).calculate(
        function
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        time_elapsed = time.time() - start_time
        print(result)
        print(time_elapsed)
