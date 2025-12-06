import numpy as np
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib.backends.backend_pdf import PdfPages


def plot_approximation(  # adapted from https://github.com/sascommunities/sas-viya-machine-learning/blob/master/gaussian_process_models/Bayesian_optimization_util.py
    gp: GaussianProcessRegressor,
    approx: Axes,
    x_target: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_next: float,
    bounds: list[float] | tuple[float, float],
):
    """_Plot approximation_

    Args:
        gp (GaussianProcessRegressor): _Gaussian Processor model_
        approx (Axes): _Plotlib axis_
        x_target (np.ndarray): _Randomly generated targets for x-axis_
        y_target (np.ndarray): _Results from objective function using x_target_
        x_obs (np.ndarray): _Values predicted by gaussian processor_
        y_obs (np.ndarray): _Results from objective function using values from x_obs_
        x_next (float): _Predicted next x value/s_
        bounds (list[float] | tuple[float, float]): _Plot bounds_
    """
    mu, std = gp.predict(x_target.reshape(-1, 1), return_std=True)
    approx.fill_between(
        x_target.ravel(),
        mu.ravel() + 1 * std,
        mu.ravel() - 1 * std,
        alpha=0.1,
        label=r"$\mu \pm \sigma$",
    )
    approx.plot(x_target, mu, "b-", lw=1, label="Prediction")
    approx.plot(x_obs, y_obs, "kx", mew=3, label="Observations")

    approx.set_xlim(bounds[0] - 0.05, bounds[1] + 0.05)
    approx.set_ylim((None, None))
    approx.set_ylabel("f(x)", fontdict={"size": 20})
    approx.set_xlabel("x", fontdict={"size": 20})

    approx.axvline(x=x_next, ls="--", c="k", lw=1, label="Next sampling location")
    approx.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)


def plot_acquistion(
    x_target: np.ndarray,
    acquisition_values: np.ndarray,
    acq: Axes,
    x_next: float,
    bounds: list[float] | tuple[float, float],
):
    """_Plot acquistion values_

    Args:
        x_target (np.ndarray): _Randomly generated targets for x-axis_
        acquisition_values (np.ndarray): _Acquisition values using current model_
        acq (Axes): _Plotlib axis_
        x_next (float): _Predicted next x value/s_
        bounds (list[float] | tuple[float, float]): _Plot bounds_
    """
    acq.set_xlim(bounds[0] - 0.05, bounds[1] + 0.05)
    acq.plot(x_target, acquisition_values, "r-", lw=1, label="Acquisition function")
    acq.axvline(x=x_next, ls="--", c="k", lw=1, label="Next sampling location")
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)


class plot_gp:
    def __init__(
        self,
        gp: GaussianProcessRegressor,
        plot_filename: str,
        bounds: tuple[float, float] | list[float],
    ):
        self._gp = gp
        self._pdf_obj = PdfPages(plot_filename)
        self._bounds = bounds

        if len(bounds) != 2:
            raise ValueError("Bounds must be only two values!")

    def __enter__(self):
        self._pdf_obj.__enter__()
        return self

    def __exit__(self, exc_type, exc_val: BaseException | None, exc_tb):
        self._pdf_obj.__exit__(exc_type, exc_val, exc_tb)
        if exc_val is not None:
            raise exc_val

    def plot(
        self,
        iteration: int,
        x_target: np.ndarray,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        x_next: float,
        acq_values: np.ndarray,
    ):
        """_Plot current iteration_

        Args:
            iteration (int): _Iteration number_
            x_target (np.ndarray): _Randomly generated targets for x-axis_
            x_obs (np.ndarray): _Values predicted by gaussian processor_
            y_obs (np.ndarray): _Results from objective function using values from x_obs_
            x_next (float): _Predicted next x value/s_
            acq_values (np.ndarray): _Acquisition values using current model_
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        approx = plt.subplot(gs[0])
        acq = plt.subplot(gs[1])

        step_title = "Iterations" if iteration > 1 else "Iteration"
        fig.suptitle(
            f"Gaussian Process and Utility Function After {iteration} {step_title}",
            fontsize=30,
        )

        plot_approximation(
            gp=self._gp,
            approx=approx,
            x_target=x_target,
            x_obs=x_obs,
            y_obs=y_obs,
            x_next=x_next,
            bounds=self._bounds,
        )

        plot_acquistion(
            x_target=x_target,
            acquisition_values=acq_values,
            acq=acq,
            x_next=x_next,
            bounds=self._bounds,
        )

        plt.tight_layout()
        self._pdf_obj.savefig()
        plt.close()
