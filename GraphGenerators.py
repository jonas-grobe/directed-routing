import math
from typing import Any, Callable, Dict
import random
import networkx as nx
import numpy as np
import fnss
from util import require_args


class GraphGenerator:
    """
    This class generates a graph using a specified method and keyword arguments.
    """

    def __init__(self, method: Callable[[dict[str, Any]], tuple[int, int, nx.Graph | nx.DiGraph]], **kwargs):
        """
        Constructor
        @param method - the method to be used for graph generation
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

    def generate(self) -> tuple[int, int, nx.Graph | nx.DiGraph]:
        """
        Generates a graph using method and kwargs. Initialises node["attr"] and edge["attr"] with 0 and edge["failed"] with False
        @return A tuple containing the start node, end node, and the generated graph.
        """
        s, d, g = self.method(**self.kwargs)
        nx.set_edge_attributes(g, "0", "attr")
        nx.set_node_attributes(g, "0", "attr")
        nx.set_edge_attributes(g, False, "failed")
        g.nodes[s]["attr"] = "s"
        g.nodes[d]["attr"] = "d"
        g.graph["root"] = d
        return s, d, g


def graph_gen(func: Callable[[dict[str, Any]], tuple[int, int, nx.Graph | nx.DiGraph]]) -> Callable[[dict[str, Any]], GraphGenerator]:
    """
    Decorator. Embeds the function into a GraphGenerator-object for use in Evaluation.eval()
    @param func - the function to generate a graph with
    @return a new function that returns a GraphGenerator-object
    """
    def wrapper(**kwargs):
        return GraphGenerator(func, **kwargs)
    return wrapper


@graph_gen
@require_args(["n", "p", "directed"])
def erdos_renyi(**kwargs) -> tuple[int, int, nx.DiGraph]:
    """
    Generates a random directed Erdos-Renyi graph with a given number of nodes and probability of edge creation.
    @param **kwargs - n: number of nodes, p: probability of creation for each possible edge, directed: output directed/undirected graph
    @return a tuple containing the source node, destination node, and the generated graph.
    """
    n = kwargs.get("n")
    p = kwargs.get("p")
    g = nx.erdos_renyi_graph(n, p, directed=kwargs.get("directed"))
    nodes = list(g.nodes)
    s, d = random.sample(nodes, 2)
    return s, d, g

@graph_gen
@require_args(["n", "area", "min_range", "max_range"])
def wireless(**kwargs) -> tuple[int, int, nx.DiGraph]:
    """
    Simulates randomly placed wireless nodes with variable sending ranges, resulting in a directed graph.
    @param **kwargs - n: number of nodes, area: area width = height, min/max_range: Every node has a range in [min_range, max_range]
    """
    n = kwargs.get("n")
    area = kwargs.get("area")
    send_range = (kwargs.get("min_range"), kwargs.get("max_range"))
    g = nx.DiGraph()
    
    for node in range(n):
        x = random.uniform(0, area)
        y = random.uniform(0, area)
        g.add_node(node, pos=(x, y), sending_range=random.uniform(*send_range))
    
    nodes = list(g.nodes)

    for u in nodes:
        for v in nodes:
            if u != v:
                pos_u = g.nodes[u]['pos']
                pos_v = g.nodes[v]['pos']
                distance = math.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)
                if distance <= g.nodes[u]['sending_range']:
                    g.add_edge(u, v)
    
    s, d = random.sample(nodes, 2)
    return s, d, g

@graph_gen
@require_args(["n", "d"])
def random_undirected(**kwargs) -> tuple[int, int, nx.Graph]:
    """
    Generates a random graph with a given number of nodes and node degrees.
    @param **kwargs - n: number of nodes, d: degree of each node. n*d must be even!
    @return a tuple containing the source node, destination node, and the generated graph.
    """
    n = kwargs.get("n")
    d = kwargs.get("d")
    g = nx.random_regular_graph(d, n, seed=None)
    nodes = list(g.nodes)
    s, d = random.sample(nodes, 2)
    return s, d, g


@graph_gen
@require_args(["n", "d", "p"])
def random_directed(**kwargs) -> tuple[int, int, nx.DiGraph]:
    """
    Generates a random directed graph with a given number of nodes and node degrees.
    @param **kwargs - n: number of nodes, d: degree of each node. n*d must be even!, p: fraction of edges to be directed in a random direction
    @return a tuple containing the source node, destination node, and the generated graph.
    """
    n = kwargs.get("n")
    d = kwargs.get("d")
    factor = kwargs.get("p")
    s, d, g = random_undirected(d=d, n=n).generate()
    l = list(g.edges)
    dir_edges = random.sample(l, int(len(l)*factor))
    g_dir = g.to_directed()
    for edge in dir_edges:
        g_dir.remove_edge(edge[0], edge[1])
    return s, d, g_dir


@graph_gen
@require_args(["filename"])
def rocketfuel(**kwargs) -> tuple[int, int, nx.Graph]:
    """
    Loads a graph from the rocketfuel project and chooses two random nodes as source and destination.
    @param **kwargs - filename: full path to the graph file to be loaded
    @return a tuple containing the source node, destination node, and the loaded graph.
    """
    g = nx.Graph()
    g.add_edges_from(fnss.parse_rocketfuel_isp_map(
        kwargs.get("filename")).edges())
    nx.convert_node_labels_to_integers(g)
    s, d = random.sample(list(g.nodes), 2)
    return s, d, g


@graph_gen
@require_args(["filename"])
def topology_zoo(**kwargs) -> tuple[int, int, nx.Graph]:
    """
    Loads a graph from the topology zoo project and chooses two random nodes as source and destination.
    @param **kwargs - filename: full path to the graph file to be loaded
    @return a tuple containing the source node, destination node, and the loaded graph.
    """
    g = nx.read_graphml(kwargs.get("filename"))
    s, d = random.sample(list(g.nodes), 2)
    return s, d, g


@graph_gen
@require_args(["n", "degree_dist"])
def random_directed_var_degree(**kwargs) -> tuple[int, int, nx.DiGraph]:
    """
    Generates a random directed graph with a given number of nodes and a probability distribution of node degrees.
    @param **kwargs - n: number of nodes, degree_dist: distribution of node degrees [(degree, prob), ...]
    @return a tuple containing the source node, destination node, and the loaded graph.
    """
    n = kwargs.get("n")
    options, dist = list(zip(*kwargs.get("degree_dist")))
    g = nx.DiGraph()
    nodes = list(range(n))
    g.add_nodes_from(nodes)
    random.shuffle(nodes)
    s = nodes[0]
    d = nodes[-1]
    # Add a random path from s to d visiting all nodes
    g.add_edges_from(zip(nodes, nodes[1:]))
    # Add random outgoing edges to every node, in a number according to prob. dist.
    for node in nodes:
        node_c = nodes.copy()
        node_c.remove(node)  # No self-loops
        n_edges = np.random.choice(options, 1, dist)[0]
        if node != d:
            # This node already has an outgoing edge
            # No second edge to neighbor
            node_c.remove(list(g.neighbors(node))[0])
            n_edges -= 1
        for other in random.sample(node_c, n_edges):
            g.add_edge(node, other)
    return s, d, g


@graph_gen
def heathland_undir(**kwargs) -> tuple[int, int, nx.Graph]:
    """
    Get the graph from the heatland experiment (see Thesis). The graph contains only undirected edges.
    @return a tuple containing the source node, destination node, and the graph.
    """
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 7), (0, 13), (2, 13), (3, 13), (2, 3), (2, 6),
                     (3, 4), (4, 6), (4, 5), (2, 12), (7, 8), (7, 11), (8, 9), (9, 10), (10, 11)])
    s = random.randint(0, 12)
    return s, 13, g


@graph_gen
def heathland(**kwargs) -> tuple[int, int, nx.DiGraph]:
    """
    Get the graph from the heatland experiment (see Thesis). The graph contains directed and undirected edges.
    @return a tuple containing the source node, destination node, and the graph.
    """
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 7), (0, 13), (2, 13), (3, 13), (2, 3), (2, 6),
                     (3, 4), (4, 6), (4, 5), (2, 12), (7, 8), (7, 11), (8, 9), (9, 10), (10, 11)])
    g = g.to_directed()
    g.add_edges_from([(1, 7), (7, 2), (13, 5), (4, 1), (3, 6),
                     (6, 5), (5, 11), (11, 6), (7, 6), (12, 8), (12, 9), (12, 10)])
    s = random.randint(0, 12)
    return s, 13, g


@graph_gen
@require_args(["s", "d", "edges"])
def fixed_graph(**kwargs) -> tuple[int, int, nx.Graph]:
    """
    Generate a graph with edges 
    @param kwargs - s: source node, d: destination node, edges: list of edges, graph_type (optional,default="dir"): "dir" or "undir"
    @return a tuple containing the source node, destination node, and the graph.
    """
    if kwargs.get("graph_type", "dir") == "dir":
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_edges_from(kwargs.get("edges"))
    return kwargs.get("s"), kwargs.get("d"), g
