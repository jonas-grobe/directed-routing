import numpy as np
import networkx as nx
from RoutingAlgos.PathDiversity import annotate_graph, get_pd_single_node


def route(g: nx.Graph, s: int, d: int, ranking: str, measure: str) -> tuple[bool, int, list[int]]:
    """
    Failure-Carrying Packet routing
    @param g - Graph to route on
    @param s - Source node
    @param d - Destination node
    @param ranking - "greedy_max", "min_max", "max"
    @param measure - "length", "path_div", "degree"
    """
    fails = []
    for edge in g.edges:
        data = g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))
    current_node = s
    hops = 0
    route = [s]
    if measure == "path_div":
        annotate_graph(g, d)
    pl_cache = {}  # node -> distance to des.
    pd_cache = {}  # node -> path diversity to des.
    cached_path = None
    while current_node != d:
        if not nx.has_path(g, current_node, d): # Can't reach t, routing failed
            return False, hops, route
        neighbors = [node for _, node in g.out_edges(current_node)]
        if d in neighbors: # Shortcut - if direct edge to target is available, always choose that
            path = [current_node, d]
        elif cached_path != None and len(cached_path) > 0:
            path = (current_node, cached_path[0])
            cached_path = cached_path[1:]
        else:
            if measure == "length":
                try:
                    path = nx.shortest_path(g, current_node, d)
                    cached_path = path[2:]
                except:
                    return False, hops, route
            elif ranking == "max" or ranking == "greedy_max":
                current_dist = pl_cache[current_node] if current_node in pl_cache else nx.shortest_path_length(g, current_node, d)
                pl_cache[current_node] = current_dist
                l = []
                for node in neighbors:
                    if not nx.has_path(g, node, d):
                        continue
                    pl = pl_cache[node] if node in pl_cache else nx.shortest_path_length(g, node, d)
                    pl_cache[node] = pl
                    if measure == "path_div":
                        val = pd_cache[node] if node in pd_cache else get_pd_single_node(g, node, d)
                        pd_cache[node] = val
                    else: # method == "degree"
                        val = len(g.out_edges(node))
                    l.append((node, val, pl-current_dist))
                if len(l) == 0: # No paths available
                    return False, hops, route
                nearer = {node: deg for node, deg, length in l if length == -1}
                equal = {node: deg for node, deg, length in l if length == 0}
                farther = {node: deg for node, deg, length in l if length >= 1}
                if ranking == "greedy_max":
                    if len(nearer) != 0:
                        path = [current_node, max(nearer, key=nearer.get)]
                    elif len(equal) != 0:
                        path = [current_node, max(equal, key=equal.get)]
                    elif len(farther) != 0: # Should never happen
                        path = [current_node, max(farther, key=farther.get)]
                else: # ranking == "max"
                    if measure == "path_div":
                        current_div = pd_cache[current_node] if current_node in pd_cache else get_pd_single_node(g, current_node, d)
                        pd_cache[current_node] = current_div
                    if len(nearer) != 0:
                        candidate = max(nearer, key=nearer.get)
                        for c in equal:
                            node, val, _ = l[c]
                            if val > l[candidate][1]:
                                candidate = node
                    else:
                        candidate = max(equal, key=nearer.get)
                    path = [current_node, candidate]
            elif ranking == "max_min":
                if measure == "path_div":
                    paths = g.nodes[current_node]["paths"]
                    if len(paths) == 0:
                        return False, hops, route
                    path_vals = [sorted([g.nodes[node]["path_diversity"] for node in path[1:-1]]) for path in paths]
                    max_len = max([len(lst) for lst in path_vals])
                    path_vals = np.array([np.pad(lst, (0, max_len - len(lst)), 'constant', constant_values=2) for lst in path_vals])
                    while len(paths) != 1:
                        try:
                            max_indices = path_vals[:, 0]
                        except:
                            paths = [paths[0]] # Its a tie
                            continue
                        non_max = np.argwhere(max_indices != np.amax(max_indices)).flatten()
                        path_vals = np.delete(path_vals, 0, axis=1) # Delete column
                        for to_remove in sorted(non_max, reverse=True):
                            del paths[to_remove]
                            path_vals = np.delete(path_vals, to_remove, axis=0) # Delete row
                    path = (current_node, paths[0][1])
                elif measure == "degree":
                    paths = list(nx.all_simple_paths(g, current_node, d))
                    if len(paths) == 0:
                        return False, hops, route
                    path_vals = [sorted([len(g.out_edges(node)) for node in path[1:-1]]) for path in paths]
                    # Normalize so max. val is 1
                    max_val = max(max(l) for l in path_vals)
                    path_vals = [[v/max_val for v in l] for l in path_vals]
                    # pad with 2's
                    max_len = max([len(lst) for lst in path_vals])
                    path_vals = np.array([np.pad(lst, (0, max_len - len(lst)), 'constant', constant_values=2) for lst in path_vals])
                    while len(paths) != 1:
                        try:
                            max_indices = path_vals[:, 0]
                        except:
                            paths = [paths[0]] # Its a tie
                            continue
                        non_max = np.argwhere(max_indices != np.amax(max_indices)).flatten()
                        path_vals = np.delete(path_vals, 0, axis=1) # Delete column
                        for to_remove in sorted(non_max, reverse=True):
                            del paths[to_remove]
                            path_vals = np.delete(path_vals, to_remove, axis=0) # Delete row
                    path = (current_node, paths[0][1])
                    cached_path = paths[0][1:]

        next_edge = (path[0], path[1])
        if next_edge in fails:
            g.remove_edge(*next_edge)
            cached_path = None
            pl_cache = {}
            pd_cache = {}
            if ranking == "max_min" and measure == "path_div":
                annotate_graph(g, d)
        else:
            hops += 1
            current_node = path[1]
            route.append(current_node)
            #if hops > max_hops:  # Infinite loop - shouldn't happen
                #retun False, hops, route
    return True, hops, route