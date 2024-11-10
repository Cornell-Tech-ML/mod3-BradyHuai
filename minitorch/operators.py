"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two floating-point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The input number.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floating-point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a floating-point number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negated number.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Compare two floating-point numbers.
    Check if x is less than y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x is less than y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compare two floating-point numbers for equality.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x is equal to y, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Find the maximum of two floating-point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two floating-point numbers are close.
    Default tolerance is 1e-2

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x and y are close, False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of a floating-point number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the ReLU function of a floating-point number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of a floating-point number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential of a floating-point number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of a floating-point number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The reciprocal of x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Calculate the derivative of the log function
    with respect to its input, and times a second input.

    Args:
    ----
        x (float): The input number of the log function.
        d (float): The second input.

    Returns:
    -------
        float: The derivative of the log times a second argument.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Calculate the derivative of the reciprocal function
    with respect to its input, and times a second input.

    Args:
    ----
        x (float): The input number of the reciprocal function.
        d (float): The second input.

    Returns:
    -------
        float: The derivative of reciprocal times a second argument.

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Calculate the derivative of the ReLU function with respect to its input,
    and times a second input.

    Args:
    ----
        x (float): The input number of the ReLU function.
        d (float): The second input.

    Returns:
    -------
        float: The derivative of ReLU times a second argument.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        f (Callable[[float], float]): The function to apply.
        ls (Iterable[float]): The list to apply the function to.

    Returns:
    -------
        Iterable[float]: The list of results after applying the function to each element.

    """
    return [f(x) for x in ls]


def zipWith(
    f: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        f (Callable[[float, float], float]): The function to apply.
        ls1 (Iterable[float]): The first list.
        ls2 (Iterable[float]): The second list.

    Returns:
    -------
        Iterable[float]: The list of results after applying the function to corresponding elements of two lists.

    """
    return [f(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    f: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        f (Callable[[float, float], float]): The function to apply.
        ls (Iterable[float]): The list to reduce.
        initial (float): The initial value.

    Returns:
    -------
        float: The reduced value.

    """
    result = initial
    for x in ls:
        result = f(result, x)
    return result


# Implement for Task 0.3.


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate a list of floating-point numbers.

    Args:
    ----
        ls (Iterable[float]): The list to negate.

    Returns:
    -------
        Iterable[float]: The list of negated numbers.

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        ls1 (Iterable[float]): The first list to add
        ls2 (Iterable[float]): The second list to add

    Returns:
    -------
        Iterable[float]: The list of added numbers

    """
    return zipWith(add, ls1, ls2)


def prod(ls: Iterable[float]) -> float:
    """Multiply all numbers in a list.

    Args:
    ----
        ls (Iterable[float]): The list to multiply.

    Returns:
    -------
        float: The product of the numbers in the list.

    """
    return reduce(mul, ls, 1)


def sum(ls: Iterable[float]) -> float:
    """Sum all numbers in a list.

    Args:
    ----
        ls (Iterable[float]): The list to sum.

    Returns:
    -------
        float: The sum of the numbers in the list.

    """
    return reduce(add, ls, 0)
