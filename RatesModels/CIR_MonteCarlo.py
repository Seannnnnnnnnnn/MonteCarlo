""" implements the discretised CIR Model """
import numpy as np
import matplotlib.pyplot as plt
from random import normalvariate


def cir(r0, mu, sigma, k, T, increment=1/252):
    """
    returns a generate CIR simulation using Monte Carlo
    :param r0: initial rate
    :param mu: long term rate
    :param sigma: volatility of rate
    :param k: mean reversion rate
    :param increment: 𝚫t
    :param T: specifies model simulation period 0 < t < T
    """
    x = [r0]
    interval = np.linspace(0 + increment, T, int(T // increment))
    for _ in interval[:-1]:
        z_t = normalvariate(0, 1)
        r_prev = x[-1]

        r_t = (1 - k*increment)*r_prev + k*mu*increment + sigma*(r_prev)**0.5 * (increment)**0.5 * z_t
        x.append(r_t)
    return x


def simulate(T, increment=1/252):
    sample1 = cir(1.1, 0.7, 0.02, 0.2, T, increment)
    sample2 = cir(1.1, 0.7, 0.02, 0.9, T, increment)
    sample3 = cir(1.1, 0.7, 0.02, 1.5, T, increment)

    interval = np.linspace(0+increment, T, int(T//increment))
    plt.plot(interval, sample1, label='k=0.2')
    plt.plot(interval, sample2, label='k=0.9')
    plt.plot(interval, sample3, label='k=1.5')
    plt.title('C.I.R Model for Rates')
    plt.ylabel('Rate')
    plt.xlabel('t')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simulate(10)