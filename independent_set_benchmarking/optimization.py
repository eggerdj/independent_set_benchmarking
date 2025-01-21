# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function to optimize lambda by balancing both Hamiltonians."""

from time import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

from qiskit_optimization.applications import StableSet
from qiskit_optimization.converters import QuadraticProgramToQubo

from independent_set_benchmarking.efficient_depth_one import EfficientDepthOne
from independent_set_benchmarking.scipy_trainer import ScipyTrainer
from independent_set_benchmarking.transpile import make_ansatz


def lambda_objective(
    lambda_,
    ansatz_sub_graph,
    qp,
    history,
    all_results,
    trainers,
    evaluator,
    minimize_args,
    obj_op,
):
    """The objective to minimize."""
    # Lambda should be positive
    lambda_ = np.sqrt(lambda_**2)

    if ansatz_sub_graph is not None:
        sub_qp = StableSet(ansatz_sub_graph).to_quadratic_program()
        sub_op, _ = QuadraticProgramToQubo(penalty=lambda_).convert(sub_qp).to_ising()
        sub_op = -1 * sub_op
        singles = sub_op[sub_op.paulis.z.sum(axis=-1) == 1]
        doubles = sub_op[sub_op.paulis.z.sum(axis=-1) == 2]
        ansatz = make_ansatz(singles, doubles)
    else:
        ansatz = None

    qubo = QuadraticProgramToQubo(penalty=lambda_).convert(qp)
    cost_op, offset = qubo.to_ising()
    cost_op = -1 * cost_op
    offset = -1 * offset

    # Train the QAOA parameters by starting from those at the previous lambda
    # This leverages the outerscope variable all_results
    trainer = ScipyTrainer(minimize_args=minimize_args)
    result = trainer.train(
        cost_op, params0=all_results[-1]["optimized_params"], ansatz_circuit=ansatz
    )
    all_results.append(result)

    trainers.append(trainer)

    # This is the value of sum_i xi
    obj_value = evaluator.evaluate(
        obj_op, params=result["optimized_params"], ansatz_circuit=ansatz
    )

    # result["energy"] + offset is the cost function of the constraint - lambda sum xixj
    constraint_value = result["energy"] - obj_value

    history.append(
        (
            -min(obj_value, constraint_value),
            lambda_,
            obj_value,
            constraint_value,
        )
    )

    return -min(obj_value, constraint_value)


def optimize_lambda(graph, lambdas, ansatz_sub_graph=None, f_minimize_args=None):
    """Optimize the lambda and QAOA parameters.

    This optimization is done by putting an equal weight on both the
    objective and the constraints operator.

    Args:
        graph: The graph that defines the independent set problem.
        lambda0: The initial value for the Lagrange multiplier.
        ansatz_sub_graph: A sub-graph of the original instance, e.g., obtained from
            a SWAP strategy simplification, from which an ansatz QAOA cost operator
            circuit will be generated. If None is given then we use the full problem.
    """
    start = time()

    # Perform a scan pre-optimization. This is needed to ensure that the optimization
    # stays ontop of the global beta and gamma optimum.
    e_cost_op, e_obj, e_cost_op2, all_vals = lambda_scan_optimize(
        lambdas,
        graph,
        ansatz_sub_graph=ansatz_sub_graph,
    )

    results = {
        "scan_optimization": {
            "cost_op_energy": e_cost_op,
            "lambda_cost_op_energy": e_cost_op2,
            "obj_energy": e_obj,
            "all_scan_data": all_vals,
        }
    }

    idx = np.argmin(abs(np.array(e_cost_op2) - np.array(e_obj)))
    lambda0 = lambdas[idx]

    # The params0 is contained in all_results.
    # Here, take the scipy optimization with the lambda closest to the optimal one.
    all_results = [all_vals[idx][0]]
    qp = StableSet(graph).to_quadratic_program()

    # Create the objective operator, i.e., the part without the penalty.
    # Note: Qiskit opt. assumes minimization but the training pipeline uses maximization.
    qubo_obj = QuadraticProgramToQubo(penalty=0).convert(qp)
    obj_op, obj_offset = qubo_obj.to_ising()
    obj_op = -1 * obj_op
    obj_offset = -1 * obj_offset

    # Find good values of beta and gamma for lambda0
    # This is done with a depth one scan followed by a Scipy optimization
    qubo0 = QuadraticProgramToQubo(penalty=lambda0).convert(qp)
    cost_op0, _ = qubo0.to_ising()
    cost_op0 = -1 * cost_op0

    evaluator = EfficientDepthOne()

    minimize_args = {"options": {"maxiter": 100, "rhobeg": 0.02}}
    trainer = ScipyTrainer(minimize_args=minimize_args)

    history, trainers = [], [trainer]

    if f_minimize_args is None:
        f_minimize_args = {
            "method": "COBYLA",
            "options": {"maxiter": 100, "rhobeg": 0.02},
        }

    result = minimize(
        lambda_objective,
        [lambda0],
        args=(
            ansatz_sub_graph,
            qp,
            history,
            all_results,
            trainers,
            evaluator,
            minimize_args,
            obj_op,
        ),
        **f_minimize_args,
    )

    results["scipy"] = {
        "scipy_result": result,
        "all_results": all_results,
        "history": history,
        "trainers": trainers,
    }

    results["train_duration"] = time() - start

    return results


def qaoa_train(
    graph,
    lambda_,
    params0,
    ansatz_sub_graph=None,
):
    """Perform an energy scan of the landscape."""
    qp = StableSet(graph).to_quadratic_program()

    qubo_obj = QuadraticProgramToQubo(penalty=0).convert(qp)
    obj_op, obj_offset = qubo_obj.to_ising()
    obj_op, obj_offset = -1 * obj_op, -1 * obj_offset

    # Find good values of beta and gamma for lambda0
    # This is done with a depth one scan followed by a Scipy optimization
    cost_op = -1 * QuadraticProgramToQubo(penalty=lambda_).convert(qp).to_ising()[0]

    if ansatz_sub_graph is not None:
        sub_qp = StableSet(ansatz_sub_graph).to_quadratic_program()
        sub_op, _ = QuadraticProgramToQubo(penalty=lambda_).convert(sub_qp).to_ising()
        sub_op = -1 * sub_op
        singles = sub_op[sub_op.paulis.z.sum(axis=-1) == 1]
        doubles = sub_op[sub_op.paulis.z.sum(axis=-1) == 2]
        ansatz = make_ansatz(singles, doubles)
    else:
        ansatz = None

    minimize_args = {"options": {"maxiter": 100, "rhobeg": 0.02}}
    trainer = ScipyTrainer(minimize_args=minimize_args)
    res_scipy_cost_op = trainer.train(
        cost_op,
        params0=params0,
        ansatz_circuit=ansatz,
    )

    energy_obj_op = EfficientDepthOne().evaluate(
        obj_op,
        res_scipy_cost_op["optimized_params"],
        ansatz_circuit=ansatz,
    )

    return res_scipy_cost_op, energy_obj_op


def lambda_scan_optimize(lambdas, graph, ansatz_sub_graph=None):
    """Optimize lambda by scanning it.

    This is necessary to ensure that we keep the global optimum. The global
    optimum is easy to find when lambda is close to zero since we only have the
    objective operator. The optimum for the objective is the |1> state which
    is produced by the angles (pi/4, pi/2) for (beta, gamma).
    """
    energies_cost_op, energies_obj, energies_cost_op2 = [], [], []
    params0 = (np.pi / 4, np.pi / 2)
    all_vals = []

    for lamb in lambdas:
        res_scipy_cost_op, energy_obj_op = qaoa_train(
            graph, lamb, params0=params0, ansatz_sub_graph=ansatz_sub_graph
        )
        all_vals.append((res_scipy_cost_op, energy_obj_op))

        energies_cost_op.append((res_scipy_cost_op["energy"] - energy_obj_op) / lamb)
        energies_cost_op2.append(res_scipy_cost_op["energy"] - energy_obj_op)
        energies_obj.append(energy_obj_op)

        params0 = res_scipy_cost_op["optimized_params"]

    return energies_cost_op, energies_obj, energies_cost_op2, all_vals


def plot_lambda_optimization(results, lambdas_scan, trainer_idx=-1):
    """Make a plot of the optimization of lambda.

    Args:
        history: The `history` return argument of `optimize_lambda`.
        trainers: List of the QAOA parameter trainers. This allows us to make
            a few sanity checks on the QAOA parameter optimization.
    """
    history = results["scipy"]["history"]
    trainers = results["scipy"]["trainers"]

    fig, axs = plt.subplots(3, 2, figsize=(9, 9))

    axs = axs.flatten()

    # Plot of the energies
    axs[0].plot(
        lambdas_scan,
        results["scan_optimization"]["obj_energy"],
        label=r"Objective $\langle H_\text{obj}\rangle^\star$",
        marker="o",
        mec="k",
    )

    math_str = r"$\lambda^{-1}\left[E_\text{QAOA}^\star - \langle H_\text{obj}\rangle^\star\right]$"
    axs[0].plot(
        lambdas_scan,
        results["scan_optimization"]["cost_op_energy"],
        label=r"Cost op. " + math_str,
        marker="o",
        mec="k",
    )
    axs[0].plot(
        lambdas_scan,
        results["scan_optimization"]["lambda_cost_op_energy"],
        label=r"Cost op. $E_\text{QAOA}^\star - \langle H_\text{obj}\rangle^\star$",
        marker="o",
        mec="k",
    )
    axs[0].scatter(
        [float(hist[1]) for hist in history],
        [float(hist[2]) for hist in history],
        color="C0",
        marker="*",
        edgecolor="k",
    )
    axs[0].scatter(
        [float(hist[1]) for hist in history],
        [float(hist[3]) for hist in history],
        color="C2",
        marker="*",
        edgecolor="k",
    )
    axs[0].legend()
    axs[0].set_xlabel(r"Lagrange multiplier $\lambda$")
    axs[0].set_ylabel("Energy")

    # Plot of the QAOA parameters
    axs[1].plot(
        lambdas_scan,
        [
            res[-2]["optimized_params"][0]
            for res in results["scan_optimization"]["all_scan_data"]
        ],
        label=r"$\beta$",
    )
    axs[1].plot(
        lambdas_scan,
        [
            res[-2]["optimized_params"][1]
            for res in results["scan_optimization"]["all_scan_data"]
        ],
        label=r"$\gamma$",
    )
    axs[1].set_xlabel(r"Lagrange multiplier $\lambda$")
    axs[1].set_ylabel("QAOA parameter value")
    axs[1].legend()

    # Plot of the objective
    axs[2].plot([-hist[0] for hist in history])

    # Plot of the lagrange multiplier
    axs[3].plot([np.sqrt(hist[1] ** 2) for hist in history])
    axs[3].text(
        len(history) * 0.5,
        history[-1][1][0] * 1.01,
        f"Optimal penalty {history[-1][1][0]:.2f}",
    )

    # Plot of the components of the objective
    axs[4].plot(
        [hist[2] for hist in history],
        label=r"Objective energy: $\langle H_\text{obj.}\rangle$",
    )
    axs[4].plot(
        [hist[3] for hist in history],
        label=r"Constraint energy: $\lambda\langle H_\text{const.}\rangle$",
        color="C2",
    )
    axs[4].legend()

    # Check plot of the trainers
    lbl_lamb = float(results["scipy"]["history"][trainer_idx][1])
    trainers[trainer_idx].plot(
        axis=axs[5],
        fig=fig,
        color="firebrick",
        label=r"$(\beta,\gamma)$ optimization at $\lambda=$" + f"{lbl_lamb:.2f}",
    )
    axs[5].set_ylabel(r"Energy $\langle H_C\rangle$")
    axs[5].set_xlabel(r"Iteration over $(\beta,\gamma)$")
    axs[5].legend()
    fig.tight_layout()
    for idx, lbl in enumerate(
        [r"Objective $F(\lambda)$", r"Lagrange multiplier $\lambda$", "Energy"]
    ):
        axs[idx + 2].set_xlabel(r"Iteration over $\lambda$")
        axs[idx + 2].set_ylabel(lbl)

    fig.tight_layout()

    return fig, axs
