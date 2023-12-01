# Based on https://github.com/oliver306/TREE/blob/master/src/kResiliencePaths.py
# - Modified to work on directed graphs

import networkx as nx
import copy


def kResiliencePaths(g: nx.Graph, s: int, d: int) -> nx.Graph:
    """
    Build routes
    @param g - graph to build the routes on
    @param s - source node
    @param d - destination node
    @return graph, EDPs marked with edge["attr"]
    """
    fails = []
    for edge in g.edges:
        data = g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))

    g_copy = copy.deepcopy(g)
    # "attr" 0 -> unused
    nx.set_edge_attributes(g_copy, 0, "attr")

    try:
        edge_disjoint_paths = list(nx.edge_disjoint_paths(g_copy, s, d))
        edge_disjoint_paths.sort(key=lambda x: len(x), reverse=False)
    except:
        return g_copy  # No path to destination, return unmodified graph -> routing will fail

    # Every path gets its own "attr" number, starting with 1
    no_path = 1
    for path in edge_disjoint_paths:
        for i in range(0, len(path)-1):
            g_copy[path[i]][path[i+1]]["attr"] = no_path
        no_path += 1

    return g_copy


def routing_kResiliencePaths(marked_g: nx.Graph, s: int, d: int) -> tuple[bool, int, list[tuple[int, int]]]:
    """
    Route
    @param g - graph to build the routes on
    @param s - source node
    @param d - destination node
    @return found (bool), hops, route (as list of edges)
    """
    result = True  # we will find out if the graph can guarantee resiliency against failures or not
    hops = 0

    # we remove the failed edges and the edges which do not belong to any structure (i.e. with the '0' attribute) from the graph
    for e in list(marked_g.edges):
        e_data = marked_g.get_edge_data(e[0], e[1])
        if e_data.get("failed") or e_data.get("attr") == 0:
            marked_g.remove_edge(e[0], e[1])

    # we will test if we can still reach the destination node; for this we will analyze each subgraph (path) until we will find the destination node
    paths_attributes = []
    for _, _, data in marked_g.edges(data=True):
        if data['attr'] not in paths_attributes:
            paths_attributes.append(data['attr'])

    # continue generating path attributes until the destination node will be reached
    attributes_gen = (attr for attr in paths_attributes if result)
    route = []
    found = False
    for attr in attributes_gen:
        T = nx.Graph()  # we reconstruct the path
        T.add_edges_from(
            [(u, v) for u, v, d in marked_g.edges(data=True) if d['attr'] == attr])
        if s in list(T.nodes):
            # we obtain a DFS order of the edges of the current path
            # the edges are labeled with the attributes ‘forward’, ‘nontree’, and ‘reverse’
            # 'nontree'-labeled edges should be removed from the dfs_edge_order_list (because the edge is not in the DFS tree)
            dfs_edge_order_list = list(nx.dfs_labeled_edges(T, s))
            for n1, n2, label in dfs_edge_order_list:
                if label == "nontree" or n1 == n2:  # we also remove self-loops
                    dfs_edge_order_list.remove((n1, n2, label))
            dfs_edges_gen = (
                edge_dfs for edge_dfs in dfs_edge_order_list if result)
            hops_current_path = 0
            # we check if we can still reach the destination node by using the current path
            for edge_dfs in dfs_edges_gen:
                # Get right order. first = current node, second = next node
                if edge_dfs[2] == "reverse":
                    first = edge_dfs[1]
                    second = edge_dfs[0]
                else:
                    first = edge_dfs[0]
                    second = edge_dfs[1]
                route.append((first, second))
                if not marked_g.has_edge(first, second):
                    # Tried to use non-existent edge, happens when routing on directed graphs
                    return False, hops + hops_current_path, route[:-1]
                hops_current_path += 1
                # destination reached
                if str(second) == str(d) or str(first) == str(d):
                    result = False  # Breaks the generators and exits the loop
                    found = True

            hops += hops_current_path

    return found, hops, route
