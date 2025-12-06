from typing import TYPE_CHECKING, Callable, cast
from tqdm import tqdm, trange
from scipy.stats.qmc import Sobol
import numpy as np
from loguru import logger
from contextlib import nullcontext

from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

from .acquisition import weighted_expected_improvement
from .plot_graph import plot_gp

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel


__all__ = ["optimize"]


def _get_next_x(
    gp: GaussianProcessRegressor,
    f_best: float,
    bounds: list[float] | tuple[float, float],
) -> float:
    """_summary_

    Args:
        gp (GaussianProcessRegressor): Gaussian Process Regressor function
        f_best (float): minimum observed f(x) so far
        bounds (list[float] | tuple[float, float]): bounds for x

    Returns:
        _float_: predicted next x
    """
    x_next = 0.0
    best_acq_value = np.inf
    for _ in range(10):
        opt_res = minimize(
            fun=lambda x: -1
            * weighted_expected_improvement(
                f_best,
                *gp.predict(x.reshape(1, -1), return_std=True),
            ),
            x0=np.random.uniform(*bounds),
            bounds=[bounds],
            options={"maxfun": 10},  # run WEI 10 times
            method="L-BFGS-B",
        )

        # Update if we found a better value
        y_out = opt_res.fun
        if y_out < best_acq_value:
            x_next = cast(float, opt_res.x)
            best_acq_value = y_out
    return x_next


def optimize(
    objective_func: Callable[[float, int], float],
    iterations: int = 10,
    n_init_samples: int = 3,
    kernel: "Kernel | None" = None,
    plot_graph_location: str | None = None,
):
    """_summary_

    Args:
        objective_func (Callable[[float], float]): _Function to optimize_
        n_init_samples (int): _Number of initial samples to generate_
        iterations (int, optional): _Total number of iterations to run
        inclusive of init samples_. Defaults to 10.
        kernel (Kernel | None, optional): _Kernel to use for GausianProcess;
        if not provided, defaults to Matern_. Defaults to None.
        plot_graph_location (str | None, optional): _Where to save plotted
        graph. If not provided, will not generate a graph_. Defaults to None.
    """

    total_iterations = iterations - n_init_samples
    bounds = [1e-3, 1.0]  # normalized values from (0,1]
    x_random_points = np.array([])
    random_points = 200
    if plot_graph_location:
        x_random_points = np.linspace(*bounds, random_points)
    if kernel is None:
        kernel = C(1.0) * Matern(nu=2.5)
    sobol = Sobol(d=1)

    logger.info(f"Generating {n_init_samples} initial sample points")

    x_samples = sobol.random(n=n_init_samples)
    y_samples = np.array(
        [
            objective_func(float(x), 10)
            for x in tqdm(x_samples, desc="Generating initial points")
        ]
    )

    gp = GaussianProcessRegressor(kernel=kernel)

    with (
        plot_gp(gp=gp, plot_filename=plot_graph_location, bounds=bounds)
        if plot_graph_location
        else nullcontext() as plotter
    ):
        for iter in trange(total_iterations, desc="Running bayesian optimization"):
            # reshape since predict and fit only accept matrix like,
            # MxN where M is number of samples and N number of
            # features we're only optimizing for one hyperparameter
            # hence this is effectively M x 1
            x_samples_reshaped = np.array(x_samples).reshape(-1, 1)
            y_samples_reshaped = np.array(y_samples).reshape(-1, 1)
            # Update Gaussian process with existing samples
            gp.fit(x_samples_reshaped, y_samples_reshaped)

            # Evaluate acquisition function at each point and find the min
            f_best = np.min(y_samples)

            x_next = _get_next_x(gp=gp, f_best=f_best, bounds=bounds)

            # Evaluate objective function at next sample and add it to existing samples

            y_next = objective_func(float(x_next), 10)

            if plotter:
                plotter.plot(
                    x_target=x_random_points,
                    x_next=x_next,
                    acq_values=weighted_expected_improvement(
                        f_best,
                        *gp.predict(x_random_points.reshape(-1, 1), return_std=True),
                    ),
                    x_obs=x_samples,
                    y_obs=y_samples,
                    iteration=iter + 4,
                )
            x_samples = np.append(x_samples, x_next)
            y_samples = np.append(y_samples, y_next)
