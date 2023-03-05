""" RatesModels Model implemented using the Euler Maruyana Sceheme """
import numpy as np
import matplotlib.pyplot as plt
from random import normalvariate


def vasicek(r0, mu, sigma, k, T, increment=1/252):
    """
    returns a generate Vasicek simulation using Monte Carlo
    :param r0: initial rate
    :param mu: long term rate
    :param sigma: volatility of rate
    :param k: mean reversion rate
    :param increment: 𝚫t
    :param T: specifies model simulation period 0 < t < T
    """
    x = [r0]
    interval = np.linspace(0+increment, T, int(T//increment))
    for _ in interval[:-1]:
        r_prev = x[-1]
        z_t = normalvariate(0, 1)
        r_t = r_prev + k*(mu - r_prev)*increment + sigma*(increment)**0.5 * z_t
        x.append(r_t)
    return x


def vasicek_improved(r0, mu, sigma, k, T, increment=1/252):
    """
    returns a generate RatesModels simulation using Monte Carlo
    :param r0: initial rate
    :param mu: long term rate
    :param sigma: volatility of rate
    :param k: mean reversion rate
    :param increment: 𝚫t
    :param T: specifies model simulation period 0 < t < T
    """
    x = [r0]
    interval = np.linspace(0+increment, T, int(T//increment))
    dwt = sigma*(increment)**0.5
    for _ in interval[:-1]:
        r_prev = x[-1]
        z_t = normalvariate(0, 1)
        r_t = r_prev + k*(mu - r_prev)*increment + dwt * z_t
        x.append(r_t)
    return x


def simulate(T, increment=0.1):
    """ simulates several paths """
    sample1 = vasicek_improved(0, 0.7, 0.02, 0.2, T, increment)
    sample2 = vasicek_improved(0, 0.7, 0.02, 0.9, T, increment)
    sample3 = vasicek_improved(0, 0.7, 0.02, 1.5, T, increment)

    interval = np.linspace(0+increment, T, int(T//increment))
    plt.plot(interval, sample1, label='k=0.2')
    plt.plot(interval, sample2, label='k=0.9')
    plt.plot(interval, sample3, label='k=1.5')
    plt.title('Vasicek Model for Rates')
    plt.ylabel('Rate')
    plt.xlabel('t')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simulate(10)
