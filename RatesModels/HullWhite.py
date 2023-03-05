""" Implements the discretised Hull-White One factor model """
import numpy as np
import matplotlib.pyplot as plt
from random import normalvariate
from typing import Callable
from math import sin


def hull_white_one_factor(r0, theta: Callable, sigma, k, T, increment=1/252):
    """
    returns a generate CIR simulation using Monte Carlo
    :param r0: initial rate
    :param theta: function describing interest rate term structure
    :param sigma: volatility of rate
    :param k: mean reversion rate
    :param increment: 𝚫t
    :param T: specifies model simulation period 0 < t < T
    """
    x = [r0]
    theta_prev = theta(0)      # variable to keep track of most previous value of theta
    interval = np.linspace(0 + increment, T, int(T // increment))
    for t in interval[:-1]:
        z_t = normalvariate(0, 1)
        r_prev = x[-1]
        r_t = theta_prev * increment + (1 - k*increment)*r_prev + sigma*increment**0.5 * z_t
        theta_prev = theta(t)  # update theta
        x.append(r_t)
    return x


def simulate(T, increment=0.1):
    """ simulates several paths """
    sample1 = hull_white_one_factor(1.1, lambda t: 1/(t+1), 0.07, 10, T, increment)
    sample2 = hull_white_one_factor(1.1, lambda t: 1/(t+1), 0.07, 0.9, T, increment)
    sample3 = hull_white_one_factor(1.1, lambda t: 1/(t+1), 0.07, 1.5, T, increment)

    interval = np.linspace(0+increment, T, int(T//increment))
    plt.plot(interval, sample1, label='k=10')
    plt.plot(interval, sample2, label='k=0.9')
    plt.plot(interval, sample3, label='k=1.5')
    plt.title('Hull-White model for Rates')
    plt.ylabel('Rate')
    plt.xlabel('t')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simulate(10, 1/252)

