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
    def calculate(self, fun, x_start, x_end, y_start, y_end):
        pass


class SequentialStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end):
        pass


class MonteCarloStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end):
        pass


class TrapezoidStrategy(CalculationStrategy):
    def calculate(self, fun, x_start, x_end, y_start, y_end):
        pass
