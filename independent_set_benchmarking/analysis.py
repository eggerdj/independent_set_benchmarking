# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods to analyze independent set runs and samples."""

from collections import defaultdict
import networkx as nx
import numpy as np

from independent_set_benchmarking.post_process import is_feasible


def counts_to_cost_val(graph: nx.Graph, counts: dict):
    """Convert a dict of counts to a dict of objective values."""

    cost_vals = defaultdict(float)
    infeasible = defaultdict(float)
    for bit_str, count in counts.items():
        candidate = [int(x) for x in bit_str[::-1]]
        if is_feasible(candidate, graph):
            cost_vals[sum(candidate)] += count
        else:
            infeasible[sum(candidate)] += count

    return cost_vals, infeasible


def to_cdf(dist_cut: dict):
    """Convert the distribution to a CDF."""
    shots = sum(dist_cut.values())
    values = sorted(dist_cut.keys())
    cdf = None
    for val in values:
        if cdf is None:
            cdf = [dist_cut[val] / shots]
        else:
            cdf.append(dist_cut[val] / shots + cdf[-1])

    return values, cdf


def random_sampling(shots: int, graph: nx.Graph):
    """Perform random sampling as a baseline."""
    dist_cut, infeasible = defaultdict(float), defaultdict(float)
    samples = np.random.choice((0, 1), (shots, graph.order()))
    for sample in samples:
        if is_feasible(sample, graph):
            dist_cut[sum(sample)] += 1
        else:
            infeasible[sum(sample)] += 1
    return dist_cut, infeasible, samples


def average_results(results: list, plot_opts: dict = None):
    # x values might be different from one result to the next
    all_xs = set()
    for res in results:
        all_xs.update(to_cdf(res)[0])

    all_xs = sorted(list(all_xs))

    # Get the average and std for the xs
    all_ys = []
    for res in results:
        x, y = to_cdf(res)

        if y is None:
            continue

        data = {x_: y_ for x_, y_ in zip(x, y)}

        fixed_y, prev_y = [], 0
        for x_ in all_xs:
            fixed_y.append(data.get(x_, prev_y))
            prev_y = fixed_y[-1]

        all_ys.append(fixed_y)
    
    all_ys = np.array(all_ys)

    data = {"x": all_xs, "y": np.average(all_ys, axis=0), "yerr": np.std(all_ys, axis=0)}
    if plot_opts is not None:
        data.update(plot_opts)

    return data
