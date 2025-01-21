# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods to transpile the circuits for independent set."""

import copy
from typing import List
import networkx as nx

from qiskit import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qopt_best_practices.transpilation.qaoa_construction_pass import (
    QAOAConstructionPass,
)
from qopt_best_practices.transpilation import qaoa_swap_strategy_pm


def circuit_construction(
    singles, doubles, params, backend, swap_strat, edge_coloring, metadata
):
    """Method to create the ISA Circuit for the backend."""
    circuits_dict = {}

    n = len(doubles[0].paulis[0])

    # Setup a cost op with the two-qubit terms only.
    doubles_circ = QAOAAnsatz(
        doubles,
        initial_state=QuantumCircuit(n),
        mixer_operator=QuantumCircuit(n),
    )

    circuits_dict["doubles"] = doubles_circ

    # Apply the swap strategy
    my_properties = {}  # We will need this later on.

    def get_permutation(pass_, dag, time, property_set, count):
        my_properties["virtual_permutation_layout"] = property_set[
            "virtual_permutation_layout"
        ]

    config = {
        "num_layers": 1,
        "swap_strategy": swap_strat,
        "edge_coloring": edge_coloring,
        "construct_qaoa": False,
    }

    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)
    circuits_dict["tdoubles"] = tdoubles_circ

    singles_circ = QuantumCircuit(n)
    singles_circ.append(
        PauliEvolutionGate(singles, time=2 * tdoubles_circ.parameters[0]), range(n)
    )
    tsingles = transpile(singles_circ, basis_gates=["rz"])
    cost_circ = tsingles.compose(tdoubles_circ, inplace=False)
    circuits_dict["cost_circuit"] = cost_circ

    # Finally, construct the full QAOA circuit.
    construction_pass = QAOAConstructionPass(1)
    construction_pass.property_set = my_properties  # merge back-in the permutation.
    transpiled_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(cost_circ)))

    circ_to_sample = transpiled_circ.assign_parameters(params, inplace=False)

    circuits_dict["circuit_to_sample"] = circ_to_sample

    if backend is not None:
        pm1 = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            scheduling_method="alap",
        )
        circuits_dict["backend"] = pm1.run(circ_to_sample)

        metadata["parameters"] = params
        circuits_dict["backend"].metadata = metadata

    return circuits_dict


def swap_strategy_simplify(swap_strat, graph: nx.Graph) -> List[nx.Graph]:
    """Generates one new graph per layer of the swap strategy.

    This allows us to create problems that require an increasing number of
    two-qubit gates to run.
    """
    sub_graph, all_sub_graphs = nx.Graph(), []

    for node in graph.nodes():
        sub_graph.add_node(node)

    for idx in range(len(swap_strat)):
        for new_edge in swap_strat.new_connections(idx):
            if graph.get_edge_data(*new_edge) is not None:
                sub_graph.add_edge(*new_edge)

        all_sub_graphs.append(copy.deepcopy(sub_graph))

    return all_sub_graphs


def make_ansatz(singles, doubles=None) -> QuantumCircuit:
    """Create a quantum circuit that will act as ansatz.

    The output of this function can be used as intput ansatz for the train method
    of the QAOA training pipeline.
    """
    gamma = Parameter("γ")
    nqubits = len(singles[0].paulis[0])
    ansatz = QuantumCircuit(nqubits)
    ansatz.append(PauliEvolutionGate(singles, time=2 * gamma), range(nqubits))

    if doubles is not None:
        ansatz.append(PauliEvolutionGate(doubles, time=2 * gamma), range(nqubits))

    return transpile(ansatz.decompose(), basis_gates=["rz", "rzz"])
