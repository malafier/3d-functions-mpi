import math
from abc import ABC, abstractmethod


def ricker_wavelet(x, y):
    return (1 - 2 * math.pi ** 2 * (x ** 2 + y ** 2)) * math.exp(-math.pi ** 2 * (x ** 2 + y ** 2))


def rosenbrock_fun(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def schwefel_fun(x, y):
    return 418.9829 * 2 - (x * math.sin(math.sqrt(abs(x))) + y * math.sin(math.sqrt(abs(y))))


class CalculationStrategy(ABC):
    @abstractmethod
    def calculate(self, fun, x_start, x_end, y_start, y_end, n):
        pass


class SequentialTrapezoidStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end, n):
        h_x = (x_end - x_start) / n
        h_y = (y_end - y_start) / n
        result = 0
        for i in range(n):
            for j in range(n):
                x_1 = x_start + i * h_x
                x_2 = x_start + (i + 1) * h_x
                y_1 = y_start + j * h_y
                y_2 = y_start + (j + 1) * h_y

                z_1, z_2, z_3, z_4 = fun(x_1, y_1), fun(x_2, y_1), fun(x_1, y_2), fun(x_2, y_2)
                result += (z_1 + z_2 + z_3 + z_4) / 4 * h_x * h_y
        return result


class TrapezoidStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end, n):
        pass


class SequentialMonteCarloStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end, n):
        pass


class MonteCarloStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end, n):
        pass
