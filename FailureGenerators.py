import random
from typing import Any, Callable
import networkx as nx
import math
from util import require_args


class FailureGenerator:
    """
    This is a class that generates failures in a graph. It takes a method and a set of keyword arguments as input. The method is the function that generates the failures, and the keyword arguments are the arguments to that function. 
    """

    def __init__(self, method, **kwargs):
        """
        Constructor
        @param method - the method to be used for failure generation
        @param kwargs - arguments that are inserted into method with each call to generate(). They can be modified with set_kwarg()
        """
        self.method = method
        self.kwargs = kwargs

    def set_kwarg(self, **kwargs):
        """
        Update the kwargs supplied to method in every generate() call
        @param kwargs - Argument(s) to update + new values
        """
        self.kwargs.update(kwargs)

    def generate(self, g: nx.Graph) -> list[tuple[int, int]]:
        """
        Generates failures on a graph using method and kwargs.
        @param g - graph to generate failures on
        @return List of failed edges.
        """
        failed_edges = self.method(g, **self.kwargs)
        # Mark failed edges in graph with edge["failed"] = True
        for node1, node2 in failed_edges:
            g[node1][node2]["failed"] = True
        return failed_edges


def failure_gen(func: Callable[[dict[str, Any]], list[tuple[int, int]]]):
    """
    Decorator. Embeds the function into a FailureGenerator-object for use in Evaluation.eval()
    @param func - the function to generate failures with. Should return a list of failed edges and not change the graph
    @return a new function that returns a FailureGenerator-object
    """
    def wrapper(**kwargs):
        return FailureGenerator(func, **kwargs)
    return wrapper


@failure_gen
def no_failure(g: nx.Graph, **kwargs) -> list[tuple[int, int]]:
    """
    @param g - Graph to choose edges to fail from
    @return empty list
    """
    return []


@failure_gen
@require_args(["failures"])
def fixed_failure(g: nx.Graph, **kwargs) -> list[tuple[int, int]]:
    """
    @param g - Graph to choose edges to fail from
    @param **kwargs - failures: List of edges to fail
    @return list of failed edges given in kwargs.failures
    """
    return kwargs.get("failures")


@failure_gen
def random_failure(g: nx.Graph, **kwargs) -> list[tuple[int, int]]:
    """
    Let random edges fail.
    @param g - Graph to choose edges to fail from
    @param **kwargs - failure_rate: failure probability for every edge OR no_failed_edges: total number of edges to fail, undir_fail: If a->b fails, b->a fails too (default False)
    @returns list of failed edges
    """
    fail_prob = kwargs.get("failure_rate")
    undir_fail = kwargs.get("undir_fail", False)
    no_failed_edges = kwargs.get("no_failed_edges")
    edges_to_fail = int(round(len(g.edges) * fail_prob)
                        ) if fail_prob != None else no_failed_edges

    failed_edges = random.sample(list(g.edges()), edges_to_fail)
    if undir_fail and g.is_directed():
        for (u, v) in failed_edges.copy():
            if g.has_edge(v, u):
                failed_edges.append((v, u))
    return failed_edges


@failure_gen
def clustered_failure(g: nx.Graph, **kwargs) -> list[tuple[int, int]]:
    """
    Let random edges fail.
    @param g - Graph to choose edges to fail from
    @param **kwargs - failure_rate: failure probability for every edge: failure_drop: failure rate drop, undir_fail: If a->b fails, b->a fails too (default False)
    @returns list of failed edges
    """
    undir_fail = kwargs.get("undir_fail", False)
    failure_rate = kwargs.get("failure_rate") if kwargs.get(
        "failure_rate") != None else 0.6
    failure_drop = kwargs.get("failure_drop") if kwargs.get(
        "failure_drop") != None else 0.3
    d = kwargs.get("d", g.graph["root"])
    failed_edges = []
    if nx.is_directed(g):
        incidents = list(g.in_edges(d))  # Look at edges  * -> d
    else:
        incidents = list(g.edges(d))

    failurePercent = failure_rate
    while failurePercent > 0.0:
        sampleNr = math.floor(failurePercent * len(incidents))
        failed_edges.extend(random.sample(incidents, sampleNr))
        failed_edges = list(set(failed_edges))  # remove duplicates

        next_incidents = []
        for edge in incidents:
            if edge[1] == d:
                if nx.is_directed(g):
                    next_incidents.extend(list(g.in_edges(edge[0])))
                else:
                    next_incidents.extend(list(g.edges(edge[0])))
            else:
                if nx.is_directed(g):
                    next_incidents.extend(list(g.in_edges(edge[1])))
                else:
                    next_incidents.extend(list(g.edges(edge[1])))

        next_incidents = list(set(next_incidents))  # remove duplicates
        incidents = next_incidents
        failurePercent -= failure_drop
    if undir_fail and g.is_directed():
        for (u, v) in failed_edges.copy():
            if g.has_edge(v, u):
                failed_edges.append((v, u))
    return failed_edges
