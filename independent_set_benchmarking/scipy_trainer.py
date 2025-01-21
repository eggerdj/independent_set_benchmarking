# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This class is an interface to SciPy's minimize function."""

from time import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from independent_set_benchmarking.efficient_depth_one import EfficientDepthOne


class ScipyTrainer:
    """A trainer that wraps SciPy's minimize function."""

    def __init__(self, minimize_args: Optional[Dict[str, Any]] = None):
        """Initialize the trainer.

        Args:
            evaluator: An instance of `BaseEvaluator` which will evaluate the enrgy
                of the QAOA circuit.
            minimize_args: Arguments that will be passed to SciPy's `minimize`.
        """
        self._evaluator = EfficientDepthOne()

        self._minimize_args = {"method": "COBYLA"}

        minimize_args = minimize_args or {}
        self._minimize_args.update(minimize_args)

        # Energy history is saved internally at each optimization for plotting.
        self._energy_history = []

        # Parameter history is saved internally
        self._parameter_history = []

    @property
    def energy_history(self) -> List[float]:
        """Return the energy history of the last optimization run."""
        return self._energy_history

    @property
    def parameter_history(self) -> List[List[float]]:
        """Return the parameter history of the last optimization run."""
        return self._parameter_history

    def train(
        self,
        cost_op: SparsePauliOp,
        params0: List[float],
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ):
        r"""Call SciPy's minimize function to do the optimization.

        Args:
            cost_op: The cost operator :math:`H_C` of the problem we want to solve.
            params0: The initial point passed to the `minimize` function.
            mixer: A quantum circuit representing the mixer of QAOA. This allows us to
                accommodate, e.g., warm-start QAOA. If this is None, then we assume the
                standard QAOA mixer.
            initial_state: A quantum circuit the represents the initial state. If None is
                given then we default to the equal superposition state |+>.
            ansatz_circuit: The ansatz circuit in case it differs from the standard QAOA
                circuit given by :math:`\exp(-i\gamma H_C)`.
        """
        start = time()

        self._energy_history = []
        self._parameter_history = []

        def _energy(x):
            """Maximize the energy by minimizing the negative energy."""
            energy = -1 * self._evaluator.evaluate(
                cost_op=cost_op,
                params=x,
                ansatz_circuit=ansatz_circuit,
            )

            self._energy_history.append(-1 * energy)
            self._parameter_history.append(list(val for val in x))

            return energy

        result = minimize(_energy, np.array(params0), **self._minimize_args)

        result = self.standardize_scipy_result(result, params0, time() - start)
        result["energy_history"] = self._energy_history
        result["parameter_history"] = self._parameter_history

        return result

    @staticmethod
    def standardize_scipy_result(result, params0, train_duration) -> dict:
        """Standardizes results from SciPy such that it can be serialized."""
        result = dict(result)
        result["optimized_params"] = result.pop("x").tolist()
        result["energy"] = -1 * result.pop("fun")
        result["x0"] = params0
        result["train_duration"] = train_duration

        # Serialize the success bool to avoid json crashing
        if "success" in result:
            success = result["success"]
            result["success"] = f"{success}"

        return result

    def plot(
        self,
        axis: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        **plot_args,
    ):
        """Plot the energy history.

        Args:
            axis: The axis object on which to plot. If None is given then we create one.
            fig: The figure instance. If no axis are given then we create one.
            plot_args: Key word arguments that are given to axis.plot().

        Returns:
            An axis instance and figure handle. These are the inputs when given.
        """

        if axis is None or fig is None:
            fig, axis = plt.subplots(1, 1)

        plot_style = {"lw": 2, "color": "dodgerblue"}
        plot_style.update(plot_args)

        axis.plot(self._energy_history, **plot_style)

        axis.set_xlabel("Iteration number")
        axis.set_ylabel("Energy")

        return fig, axis
