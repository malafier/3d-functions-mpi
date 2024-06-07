from calculator import SequentialTrapezoidStrategy, rosenbrock_fun


if __name__ == '__main__':
    result = SequentialTrapezoidStrategy(-2, 2, -1, 3, 100).calculate(rosenbrock_fun)
    print(result)
