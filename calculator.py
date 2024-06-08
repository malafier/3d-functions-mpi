import math
from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI


def ricker_wavelet(x, y):
    return (1 - 2 * math.pi ** 2 * (x ** 2 + y ** 2)) * math.exp(-math.pi ** 2 * (x ** 2 + y ** 2))


def rosenbrock_fun(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def schwefel_fun(x, y):
    return 418.9829 * 2 - (x * math.sin(math.sqrt(abs(x))) + y * math.sin(math.sqrt(abs(y))))


def trapezoid_integral(x_start, y_start, fun, h_x, h_y, i, j):
    x_1 = x_start + i * h_x
    x_2 = x_start + (i + 1) * h_x
    y_1 = y_start + j * h_y
    y_2 = y_start + (j + 1) * h_y

    z_1, z_2, z_3, z_4 = fun(x_1, y_1), fun(x_2, y_1), fun(x_1, y_2), fun(x_2, y_2)
    return (z_1 + z_2 + z_3 + z_4) / 4 * h_x * h_y


class CalculationStrategy(ABC):
    def __init__(self, x_start, x_end, y_start, y_end, n):
        self.comm = MPI.COMM_WORLD
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.n = n

    @abstractmethod
    def calculate(self, fun):
        pass


class SequentialTrapezoidStrategy(CalculationStrategy):
    def calculate(self, fun):
        h_x = (self.x_end - self.x_start) / self.n
        h_y = (self.y_end - self.y_start) / self.n

        result = 0
        for i in range(self.n):
            for j in range(self.n):
                result += trapezoid_integral(self.x_start, self.y_start, fun, h_x, h_y, i, j)
        return result


class ParallelTrapezoidStrategy(CalculationStrategy):
    def calculate(self, fun):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        h_x = (self.x_end - self.x_start) / self.n
        h_y = (self.y_end - self.y_start) / self.n
        result = 0

        if rank == 0:
            n_per_process = self.n // size
            for i in range(1, size):
                self.comm.send([
                    self.x_start + i * n_per_process * h_x,
                    self.y_start + i * n_per_process * h_y,
                    n_per_process
                ], dest=i, tag=1)

            for i in range(n_per_process):
                for j in range(n_per_process):
                    result += trapezoid_integral(self.x_start, self.y_start, fun, h_x, h_y, i, j)

            for i in range(1, size):
                result += self.comm.recv(source=i, tag=2)

            return result
        else:
            x_start, y_start, n = self.comm.recv(source=0, tag=1)
            local_result = 0
            for i in range(n):
                for j in range(n):
                    local_result += trapezoid_integral(x_start, y_start, fun, h_x, h_y, i, j)
            self.comm.send(local_result, dest=0, tag=2)


class SequentialMonteCarloStrategy(CalculationStrategy):
    def calculate(self, fun):
        xlist = np.random.uniform(low=self.x_start, high=self.x_end, size=self.n)
        ylist = np.random.uniform(low=self.y_start, high=self.y_end, size=self.n)
        value_sum = sum([fun(x, y) for x, y in zip(xlist, ylist)])
        result = (self.x_end - self.x_start) * (self.y_end - self.y_start) / self.n * value_sum
        return result


class ParallelMonteCarloStrategy(CalculationStrategy):
    def calculate(self, fun):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        n_local = self.n // size
        if rank < self.n % size:
            n_local += 1

        xlist = np.random.uniform(low=self.x_start, high=self.x_end, size=n_local)
        ylist = np.random.uniform(low=self.y_start, high=self.y_end, size=n_local)

        value_sum_local = sum([fun(x, y) for x, y in zip(xlist, ylist)])

        value_sum = self.comm.reduce(value_sum_local, op=MPI.SUM, root=0)

        if rank == 0:
            result = (self.x_end - self.x_start) * (self.y_end - self.y_start) / self.n * value_sum
            return result
        else:
            return None
