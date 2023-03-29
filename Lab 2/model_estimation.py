import os
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt


"""Distribution type (used for plotting true distributions)"""
class DistributionType(Enum):
    GAUSSIAN = 1,
    EXPONENTIAL = 2,

"""Type and parameters information for a given distribution"""
@dataclass
class Distribution:
    type: DistributionType
    params: tuple[float, ...]

"""Factory method for Distribution with Gaussian type"""
def gaussian_distribution(mean: float, std: float) -> Distribution:
    return Distribution(type=DistributionType.GAUSSIAN, params=(mean, std))

"""Factory method for Distribution with Exponential type"""
def exponential_distribution(lambda_: float) -> Distribution:
    return Distribution(type=DistributionType.EXPONENTIAL, params=(lambda_,))


"""Method to be used by estimate_model_1d and estimate_model_2d"""
class EstimationMethod(Enum):
    GAUSSIAN = 1,
    EXPONENTIAL = 2,
    UNIFORM = 3,
    PARZEN = 4,


"""Main code for part 1 of the lab"""
def estimate_model_1d(method: EstimationMethod, data: np.ndarray, true_dist: Distribution) -> None:
    ax, x_range = _plot_true_distribution(data, true_dist)

    if method == EstimationMethod.GAUSSIAN:
        mean, std = np.mean(data), np.std(data)
        _plot_guassian(ax, x_range, mean, std, label='Estimated Distribution', color='tab:orange')
    elif method == EstimationMethod.EXPONENTIAL:
        lambda_ = 1 / np.mean(data)
        _plot_exponential(ax, x_range, lambda_, label='Estimated Distribution', color='tab:orange')
    elif method == EstimationMethod.UNIFORM:
        a, b = np.min(data), np.max(data)
        _plot_uniform(ax, x_range, a, b, label='Estimated Distribution', color='tab:orange')
    
    ax.legend()
    _save_figure('part1/{est}_estimate_for_{true}.png'.format(
        est=method.name.lower(), true=true_dist.type.name.lower()
    ))


"""Main code for part 2 of the lab"""
def estimate_model_2d(method: EstimationMethod, data: np.ndarray, true_dist: Distribution) -> None:
    pass


"""Helper function that creates correct directory for figure, saves it, and closes it"""
def _save_figure(name: str) -> None:
    path = os.path.join('plots', name)
    dir, _ = os.path.split(path)
    os.makedirs(dir, exist_ok=True)
    plt.savefig(path)
    plt.close()

"""Plots true distribution histogram and overlaid PDF"""
def _plot_true_distribution(
        data: np.ndarray,
        true_dist: Distribution) -> tuple[matplotlib.axes.Axes, tuple[float, float]]:

    # compute histogram bins
    counts, bins = np.histogram(data)
    # plot histogram with normal distribution overlaid on seperate y axis
    _, ax = plt.subplots()
    ax.hist(bins[:-1], bins, weights=counts, fc=[*matplotlib.colors.to_rgb('b'), 0.4])
    ax.set_xlabel('Measured Distance (cm)')
    ax.set_ylabel('Occurence')
    ax.set_ylim(bottom=0)

    # return seperate y axis to plot probability distribution on
    ax2 = ax.twinx()
    x_range = (bins[0], bins[-1])

    # plot true distribution
    if true_dist.type == DistributionType.GAUSSIAN:
        _plot_guassian(ax2, x_range, *true_dist.params, label='True Distribution')
    elif true_dist.type == DistributionType.EXPONENTIAL:
        _plot_exponential(ax2, x_range, *true_dist.params, label='True Distribution')
    else:
        raise ValueError("Unexpected value for type of true distribution")

    return (ax2, x_range)

"""Plots Gaussian PDF"""
def _plot_guassian(
        ax: matplotlib.axes.Axes, x_range: tuple[float, float],
        mean: float, std: float,
        label: Optional[str] = None, color: Optional[str] = 'g') -> None:
    
    assert(std > 0)

    dist_x = np.linspace(*x_range)
    dist_y = scipy.stats.norm(loc=mean, scale=std).pdf(dist_x)
            
    ax.plot(dist_x, dist_y, color, label=label)
    ax.set_ylabel('Probability Density')
    ax.set_ylim(bottom=0)

"""Plots Exponential PDF"""
def _plot_exponential(
        ax: matplotlib.axes.Axes, x_range: tuple[float, float],
        lambda_: float,
        label: Optional[str] = None, color: Optional[str] = 'g') -> None:
        
    dist_x = np.linspace(*x_range)
    dist_y = scipy.stats.expon(scale=1/lambda_).pdf(dist_x)
            
    ax.plot(dist_x, dist_y, color, label=label)
    ax.set_ylabel('Probability Density')
    ax.set_ylim(bottom=0)


"""Plots Uniform PDF"""
def _plot_uniform(
        ax: matplotlib.axes.Axes, x_range: tuple[float, float],
        a: float, b: float,
        label: Optional[str] = None, color: Optional[str] = 'g') -> None:

    assert(b > a)

    # expand x_range if necessary to include all of [a, b] 
    x_range = list(x_range)  # tuples are immutable, convert to list  
    if a <= x_range[0]:
        x_range[0] = a - (b-a)*0.05
    if x_range[1] <= b:
        x_range[1] = b + (b-a)*0.05

    dist_x = np.linspace(*x_range, num=1000)
    dist_y = scipy.stats.uniform(loc=a, scale=b-a).pdf(dist_x)
            
    ax.plot(dist_x, dist_y, color, label=label)
    ax.set_ylabel('Probability Density')
    ax.set_ylim(bottom=0)