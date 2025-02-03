import numpy as np

def f1(x: np.ndarray) -> np.ndarray:  # mse: 0.0000
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:  # mse: 22295151144311.8398
    return ((x[2] - np.log10(x[0])) - (((abs((x[2] + (x[0] + x[1]))) - np.cosh(8.44092841095036)) * (np.cosh((np.log10((x[1] + np.log10(x[0]))) + (9.360167664306434 + x[1]))) * (x[2] + (x[0] + x[1])))) - (np.log10((x[1] + x[2])) + x[2])))

def f3(x: np.ndarray) -> np.ndarray:  # mse: 323.9211
    return (((x[0] ** np.sqrt(x[0])) + (9.369353447809665 - np.sinh(x[1]))) + (0.46145314330067744 - np.sinh(x[1])))

def f4(x: np.ndarray) -> np.ndarray:  # mse: 13.6761
    return ((3.059692744122401 - np.sin(x[0])) * np.cos(x[1]))

def f5(x: np.ndarray) -> np.ndarray:  # mse: 0.0000
    return (0.6125436029273761)

def f6(x: np.ndarray) -> np.ndarray:  # mse: 0.1417
    return ((x[1] + (np.cos(np.sinh(np.log10(np.log(-4.073294203340159)))) * ((x[1] + -0.5308183642091879) - x[0]))) - np.log(0.3790412883919465))

def f7(x: np.ndarray) -> np.ndarray:  # mse: 379.9968
    return np.cosh(((np.sqrt((x[0] - (x[1] ** x[0]))) * x[0]) - ((x[1] + x[0]) + ((x[1] + x[0]) - x[1]))))

def f8(x: np.ndarray) -> np.ndarray:  # mse: 17233958.8326
    return (np.sinh((x[5] - -2.036453075415807)) * ((-2.1372214388055433 mod x[5]) + (6.890792938120018 - x[2])))

