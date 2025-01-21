# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility methods for loading, saving, etc."""

from datetime import datetime
import json
from typing import List
import networkx as nx
import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.sat_mapping import SATMapper


def load_graph(file_name):
    """Given a file name, load the graph it contains."""

    with open(file_name, "r") as file:
        lines = [line.rstrip() for line in file]

    edge_nodes = set()
    for line in lines:
        if len(line) == 0:
            continue

        if line[0] == "p":
            header = line.split(" ")
            nnodes = int(header[2])
        if line[0] == "e":
            edge = line.split(" ")
            edge_nodes.add(int(edge[1]))
            edge_nodes.add(int(edge[2]))

    if max(edge_nodes) == nnodes:
        start_idx = 1
    elif max(edge_nodes) + 1 == nnodes:
        start_idx = 0

    adj_mat = np.zeros((nnodes, nnodes))
    for line in lines[1:]:
        if len(line) > 0 and line[0] == "e":
            edge = [int(e) for e in line.split(" ")[1:]]
            adj_mat[edge[0] - start_idx, edge[1] - start_idx] = 1

    graph = nx.from_numpy_array(adj_mat)
    graph_original = nx.from_numpy_array(adj_mat)

    return graph, graph_original


def sat_map(graphs: List[nx.Graph], swap_strategy: SwapStrategy):
    """Perform SAT mapping on a list of graphs.

    Args:
        graphs: The list of graphs to SAT map. They should all have the same order.
        swap_strategy: The SWAP strategy on which to base the SAT mapping.
    """
    sat_graphs, min_layers = [], []

    for graph in graphs:
        sm = SATMapper()
        remapped_g, _, min_sat_layers = sm.remap_graph_with_sat(
            graph=graph, swap_strategy=swap_strategy
        )

        sat_graphs.append(remapped_g)
        min_layers.append(min_sat_layers)

    return sat_graphs, min_layers


def statistics(graph: nx.Graph):
    """Compute information on the given graph."""
    isolates = set(node for node in nx.isolates(graph))
    sub_graphs = (graph.subgraph(c) for c in nx.connected_components(graph))

    return {
        "nbr isolates": len(isolates),
        "sub_graphs": [g.order() for g in sub_graphs],
    }


def save_hardware_result(sampler_job, tag: str = None):
    """Save a hardware results.

    We assume that this is being called from a notebook in the
    notebook directory.
    """
    date_time = datetime.now().strftime("%Y%m%d")

    if tag is not None:
        file_name = f"{date_time}_{sampler_job.job_id()}_{tag}.json"
        metrics_name = f"{date_time}_{sampler_job.job_id()}_{tag}_metrics.json"
    else:
        file_name = f"{date_time}_{sampler_job.job_id()}.json"
        metrics_name = f"{date_time}_{sampler_job.job_id()}_metrics.json"

    data = [(res.metadata, res.data.c.get_counts()) for res in sampler_job.result()]

    with open("../results_hardware/" + file_name, "w") as fout:
        json.dump(data, fout)

    with open("../results_hardware/" + metrics_name, "w") as fout:
        json.dump(sampler_job.metrics(), fout)


def operator_to_graph(operator: SparsePauliOp, pre_factor: float = 1.0) -> nx.Graph:
    """Convert a cost operator to a graph.

    Limitations:
    * Restricted to quadratic cost operators given as sums over :math:`Z_iZ_j`.
    * Weighted quadratic cost operators are accepted and result in weighted graphs.

    Raises:
        ValueError if the operator is not quadratic.
    """
    graph, edges = nx.Graph(), []
    for pauli_str, weight in operator.to_list():
        edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]

        if len(edge) == 1:
            edges.append((edge[0], edge[0], pre_factor * np.real(weight)))
        elif len(edge) == 2:
            edges.append((edge[0], edge[1], pre_factor * np.real(weight)))
        else:
            raise ValueError(f"The operator {operator} is not Quadratic.")

    graph.add_weighted_edges_from(edges)

    return graph


def circuit_to_graph(circuit: QuantumCircuit) -> nx.Graph:
    """Convert a circuit (corresponding to a QAOA cost operator) to a graph.

    This method allows us to convert a network of rzz gates into a graph.
    We assume that for each `Rzz(2 * w * γ, i, j)` gate we have an edge between nodes
    i and j with a weight `w`. Here, `γ` is the QAOA gamma parameter.

    Assumptions:
    * The circuit only contains Rzz operations.
    * Each of the rzz gates is parameterized by gamma.

    Raises:
        ValueError if the circuit contains anything else than a Rzz gates with one parameter.
        The function also raises if a Rzz gate is present multiple times on the same qubits.
        This is designed to make the graph that we generate unambiguous.
    """
    qreg = circuit.qregs[0]
    graph, edges = nx.Graph(), []
    graph.add_nodes_from(range(len(qreg)))
    seen_edges = set()

    for inst in circuit.data:
        iop = inst.operation

        if iop.name not in ["rzz", "rz"]:
            raise ValueError(
                f"Circuit must be composed of Rz or Rzz gates only. Found {inst.operation.name}"
            )

        if len(iop.params) != 1:
            raise ValueError("The Rz/Rzz gates should have one parameter.")

        if not isinstance(iop.params[0], ParameterExpression):
            raise ValueError("The Rzz gates should have one parameter.")

        if len(inst.qubits) == 1:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[0]))

        if len(inst.qubits) == 2:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[1]))

        if edge in seen_edges:
            raise ValueError(f"Circuit contains multiple times the edge {edge}.")

        seen_edges.add(edge)
        seen_edges.add(edge[::-1])

        weight = float(iop.params[0] / next(iter(iop.params[0].parameters))) / 2.0

        edges.append((edge[0], edge[1], weight))

    graph.add_weighted_edges_from(edges)

    return graph
