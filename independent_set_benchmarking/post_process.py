# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods to run independent set classical post processing."""

from collections import defaultdict
from typing import Dict, List
import networkx as nx


def compute_violations(sample: List[int], graph: nx.Graph) -> dict:
    """Compute the number of violations of the given node.

    If the node is in the independent set then for each of its neighbors
    compute the number of neighbors that are also in the independent set.
    """
    violations = {}
    for node in graph.nodes():
        num_violataions = 0
        for neighbor in graph.neighbors(node):
            # if node == 0 we add 0 if it is 1 we add 1 if the neighbor is 1.
            if sample[neighbor] == 1:
                num_violataions += sample[node]

        violations[node] = num_violataions

    return violations


def is_feasible(sample: List[int], graph: nx.Graph):
    """Determine independent set feasibility.

    We implement this function because it is orders of magnitude faster
    then using `QuadraticProgram.is_feasible`.
    """
    return sum(compute_violations(sample, graph).values()) == 0


def greedy_post_process(counts: Dict[str, float], graph: nx.Graph):
    """Post process the counts for the independent set problem.

    Args:
        counts: A counts dictionary.
        qp: The quadratic program.
        graph: The graph of the independent set problem.
    """
    post_processed_counts = defaultdict(float)

    best_solutions, best_val = set(), 0

    for sample_bits, count in counts.items():
        sample = [int(bit) for bit in sample_bits][::-1]

        violations = compute_violations(sample, graph)

        candidates = set(graph.nodes())
        while sum(violations.values()) > 0:
            max_offender = max(*list(violations.items()), key=lambda x: x[1])[0]
            sample[max_offender] = 0
            candidates.remove(max_offender)
            violations = compute_violations(sample, graph)

        new_sample = list(val for val in sample)
        while len(candidates) > 0:
            node = next(iter(candidates))
            if new_sample[node] == 0:
                test_sample = list(val for val in new_sample)
                test_sample[node] = 1

                num_violations = sum(compute_violations(test_sample, graph).values())

                if num_violations == 0:
                    new_sample[node] = 1

            candidates.remove(node)

        # The sample is guaranteed feasible and the objective of independent set is
        # simply the number of 1s.
        obj_value = sum(new_sample)
        if obj_value == best_val:
            best_solutions.add(tuple(new_sample))
        elif obj_value > best_val:
            best_val = obj_value
            best_solutions = {tuple(new_sample)}

        post_processed_counts[sum(new_sample)] += count

    return post_processed_counts, best_solutions, best_val
