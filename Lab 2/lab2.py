from typing import Any

import scipy

import model_estimation
import sequential_discriminants


def load_data(file_no: int) -> dict[str, Any]:
    file_name = 'data/lab2_{no}.mat'.format(no=file_no)
    return scipy.io.loadmat(file_name)


def main() -> None:
    ## Part 1 ##
    a_true_dist = model_estimation.gaussian_distribution(mean=5, std=1)
    b_true_dist = model_estimation.exponential_distribution(lambda_=1)

    data_1 = load_data(1)

    for method in model_estimation.EstimationMethod:
        model_estimation.estimate_model_1d(method, data_1['a'], a_true_dist)
        model_estimation.estimate_model_1d(method, data_1['b'], b_true_dist)

    ## Part 2 ##

    ## Part 3 ##


if __name__ == '__main__':
    main()
