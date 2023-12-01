import networkx as nx
import math

LAMBDA = 1


def get_pd_single_node(g: nx.Graph, s: int, d: int):
    """
    Calulcate path diversity
    @param g - graph
    @param s - source node
    @param d - destination node
    @return path diversity of s
    """
    path_sets = []
    k = 0.0
    try:
        paths = nx.all_simple_paths(g, s, d)
    except:
        return 0  # Not connected -> treat pd as 0
    i = 0
    for path in paths:
        edges = set(list(zip(path, path[1:])))
        if i != 0:  # Skip first iteration
            k += min(path_div(edges, p) for p in path_sets)
        i = 1
        path_sets.append(edges)
    return 1 - math.exp(-1*LAMBDA*k)


def annotate_graph(g: nx.Graph, d: int):
    """
    Annotate graph with path diversity (node["path_diversity"]). Paths to d are also stored in node["paths"]
    @param g - graph
    @param d - destination node
    """
    nx.set_node_attributes(g, False, "done")
    nx.set_node_attributes(g, 0.0, "path_diversity")
    nx.set_node_attributes(g, [], "paths")
    cutoff = len(g) - 1
    for source in g.nodes:
        if source == d:
            continue
        # Taken from nx.all_simple_paths, modifications for higher efficiency when searching for all paths * -> d
        targets = {d}
        visited = dict.fromkeys([source])
        stack = [iter(g[source])]  # Children
        paths = []
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.popitem()
            elif len(visited) < cutoff:  # Not every node visited yet
                if child in visited:
                    continue
                if child == d:
                    paths.append(list(visited) + [child])
                if g.nodes[child]["done"]:
                    visited_set = set(visited)
                    for _path in g.nodes[child]["paths"]:
                        if visited_set.intersection(_path) == set():
                            paths.append(list(visited) + list(_path))
                else:
                    visited[child] = None
                    # expand stack until find all targets
                    if targets - set(visited.keys()):
                        stack.append(iter(g[child]))
                    else:
                        visited.popitem()
            else:  # len(visited) == cutoff:
                for target in (targets & (set(children) | {child})) - set(visited.keys()):
                    paths.append(list(visited) + [target])
                stack.pop()
                visited.popitem()
        g.nodes[source]["done"] = True
        g.nodes[source]["paths"] = paths
    for node in g.nodes:
        if node == d:
            continue
        path_sets = []
        k = .0
        for i, path in enumerate(g.nodes[node]["paths"]):
            edges = set(list(zip(path, path[1:])))
            if i != 0:
                k += min(path_div(edges, p) for p in path_sets)
            path_sets.append(edges)
        g.nodes[node]["path_diversity"] = 1 - math.exp(-1*LAMBDA*k)


def path_div(p0_edges, p1_edges):
    """
    Path diversity between two paths (represented as sets of edges)
    @param p0_edges - set of edges
    @param p1_edges - set of edges
    @return path diversity
    """
    p0_len = len(p0_edges)
    p1_len = len(p1_edges)

    if p0_len > p1_len:
        p0_edges, p1_edges = p1_edges, p0_edges
    return 1 - len(p0_edges.intersection(p1_edges))/p0_len
