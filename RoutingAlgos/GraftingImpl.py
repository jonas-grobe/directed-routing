# Based on https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/blob/master/extra_links.py

import sys
import networkx as nx
import numpy as np
import random
from RoutingAlgos.BonsaiImpl import GreedyArborescenceDecomposition, edmond, get_arborescence_list, get_arborescence_dict, reset_arb_attribute, TestCut
from heapq import heappush, heappop


def MaximizeFindClusters(g: nx.Graph) -> None:
    """
    Build clustered arborescences and save them in g.graph["T"]
    @param g - graph
    """
    FindClusters(g)
    g.graph["T"] = GreedyMaximalDAG(g)


def GreedyMaximalDAG(g: nx.Graph) -> list[nx.DiGraph]:
    """
    add edges to DAGs until no further edges can be added
    @param g - graph
    @return List of all arbs
    """
    not_assigned = set((u, v) for (u, v) in g.edges() if g[u][v]['attr'] == -1)
    assigned = [1]
    count = 0
    while len(assigned) > 0:
        assigned = []
        for (u, v) in not_assigned:
            dec_dict = get_arborescence_dict(g)
            for (index, arb) in dec_dict.items():
                if v in arb.nodes():
                    temp = arb.to_directed()
                    temp.add_edge(u, v)
                    if nx.is_directed_acyclic_graph(temp):
                        g[u][v]['attr'] == index
                        assigned.append((u, v))
                        break
        not_assigned.difference_update(assigned)
    return get_arborescence_list(g)


def DegreeMaxDAG(g: nx.Graph) -> None:
    """
    add edges to DAGs until no further edges can be added
    @param g - graph
    """
    reset_arb_attribute(g)
    gg = g.to_directed()
    # K is set to degree of root
    K = len(g.in_edges(g.graph['root']))
    k = K
    while k > 0:
        T = FindTreeNoContinue(gg, k)
        if T is None or len(T.edges()) == 0:
            K = K-1
            k = k-1
            continue
        for (u, v) in T.edges():
            g[u][v]['attr'] = K-k
        gg.remove_edges_from(T.edges())
        k = k-1
    g.graph["T"] = GreedyMaximalDAG(g)


def FindTreeNoContinue(g: nx.Graph, k: int) -> nx.DiGraph:
    """
    @param g - graph
    @param k - what arb to compute
    @return k^th arborescence of g computed greedily
    """
    T = nx.DiGraph()
    T.add_node(g.graph['root'])
    R = {g.graph['root']}
    dist = dict()
    dist[g.graph['root']] = 0
    # heap of all border edges in form [(edge metric, (e[0], e[1])),...]
    h = []
    preds = sorted(g.predecessors(
        g.graph['root']), key=lambda k: random.random())
    for x in preds:
        heappush(h, (0, (x, g.graph['root'])))
        # the original FindTree method continues here if k > 1
    while len(h) > 0:
        (d, e) = heappop(h)
        g.remove_edge(*e)
        if e[0] not in R and (k == 1 or TestCut(g, e[0], g.graph['root']) >= k-1):
            dist[e[0]] = d+1
            R.add(e[0])
            preds = sorted(g.predecessors(e[0]), key=lambda k: random.random())
            for x in preds:
                if x not in R:
                    heappush(h, (d+1, (x, e[0])))
            T.add_edge(*e)
        else:
            g.add_edge(*e)
    if len(R) < len(g.nodes()):
        # print(
        #    "Couldn't find next edge for tree with g.graph['root'], ", k, len(R))
        sys.stdout.flush()
    return T


def RouteDetCircSkip(s: int, d: int, g: nx.DiGraph) -> tuple[bool, int, list[tuple[int, int]]]:
    """
    @param s - source node
    @param d - destination node
    @param g - marked graph to route on
    @return fail, hops, route 
    """
    T = g.graph["T"]
    curT = 0
    detour_edges = []
    hops = 0
    switches = 0
    fails = []
    for edge in g.edges:
        data = g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))
    k = len(T)
    if k == 0:
        return (True, hops, detour_edges)
    n = max([len(T[i].nodes()) for i in range(k)])
    dist = nx.shortest_path_length(g, target=d)
    while (s != d):
        while (s not in T[curT].nodes()) and switches < k*n:
            curT = (curT+1) % k
            switches += 1
        if switches >= k*n:
            return (True, hops, detour_edges)
        nxt = list(T[curT].neighbors(s))
        if len(nxt) == 0:
            curT = (curT+1) % k
            switches += 1
            continue
        if len(nxt) == 0:
            curT = (curT+1) % k
            switches += 1
            break
        breaking = False
        # remove bad nodes from list
        len_nxt = len(nxt)
        nxt = [x for x in nxt if x in dist.keys()]
        if len(nxt) < len_nxt:
            if len(nxt) == 0:
                curT = (curT+1) % k
                switches += 1
                break
        # sort list of next hops by distance
        nxt = sorted(nxt, key=lambda ele: dist[ele])
        index = 0
        # while (nxt[index], s) in fails or (s, nxt[index]) in fails:
        while (s, nxt[index]) in fails:
            index = index + 1
            if index >= len(nxt):
                curT = (curT+1) % k
                switches += 1
                breaking = True
                break
        if not breaking:
            # if switches > 0 and curT > 0:
            # detour_edges.append((s, nxt[index]))
            detour_edges.append((s, nxt[index]))
            s = nxt[index]
            hops += 1
        if hops > n*n or switches > k*n:
            return (True, hops, detour_edges)
    if detour_edges == []:
        return (True, hops, detour_edges)
    return (False, hops, detour_edges)

# ------------------------------------------------------------------------------
# [Andrzej] The following functions are required by the FindClusters heuristic algorithm
# 1) Find all clusters in the original graph, mark the involved nodes including direct neighbors.
# 2) Find the strongly connected components within a subgraph comprising the marked nodes.
# 3) Make sure the connected components contain at least 3 nodes.
# 4) To improve local edge connectivity, remove all nodes of degree 1 from the connected components.
# 5) Find local trees in the connected components (greedy).
# 6) Assign the unused arcs of the original graph to the new local trees (existing assignments have priority).


def compute_additional_node_parameters(G: nx.Graph, mark_cluster_neighbors: bool) -> nx.Graph:
    """
    For each node, compute additional parameters
    @param G - Graph to mark
    @param mark_cluster_neighbors - Also set "marked" of nodes neighboring clusters
    @returns marked graph ("marked", "clustering_coefficient" and "distance_to_root" set for all nodes)
    """
    # Distances from all source nodes to the root node
    distances = nx.shortest_path_length(G, target=G.graph['root'])

    # Clustering coefficient of each node
    clustering_coefficients = nx.clustering(G)

    # Other parameters
    for u in G.nodes():
        G.nodes[u]['marked'] = False

    for u in G.nodes():
        G.nodes[u]['clustering_coefficient'] = clustering_coefficients[u]

        if not u in distances:
            distances[u] = len(G.edges())

        G.nodes[u]['distance_to_root'] = distances[u]

        # Mark nodes based on the following criteria
        if clustering_coefficients[u] > 0.0:
            G.nodes[u]['marked'] = True
            if mark_cluster_neighbors:
                for v in G[u]:
                    G.nodes[v]['marked'] = True

    return G


def return_subgraph_with_marked_nodes(G: nx.Graph) -> nx.DiGraph:
    """
    @param G - marked graph
    @return a subgraph of 'G' based on the marked nodes
    """
    marked_nodes = list()

    for u in G.nodes():
        if 'marked' in G.nodes[u].keys() and G.nodes[u]['marked'] == True:
            marked_nodes.append(u)

    Gm = nx.DiGraph(G.subgraph(marked_nodes))

    # Remove nodes which have been included in the subgraph, but had not been marked
    nodes_to_remove = list()
    for u in Gm.nodes():
        if not 'marked' in Gm.nodes[u].keys() or Gm.nodes[u]['marked'] == False:
            nodes_to_remove.append(u)

    Gm.remove_nodes_from(nodes_to_remove)

    return Gm


def return_nodes_of_degree_one(G: nx.Graph) -> list[int]:
    """
    @param G - Graph
    @return the list of nodes having degree 1
    """
    deg_one_nodes = list()

    for u in G.nodes():
        if len(G[u]) == 1:
            deg_one_nodes.append(u)

    return deg_one_nodes


def return_gw_node_towards_the_global_root(G: nx.Graph) -> int:
    """
    @param G - Graph
    @return the node which is the closest to the destination
    """
    gw = -1
    distance = -1

    for u in G.nodes():
        if distance == -1 or G.nodes[u]['distance_to_root'] < distance:
            gw = u
            distance = G.nodes[u]['distance_to_root']

    return gw


def FindClusters(g: nx.Graph) -> list[nx.DiGraph]:
    """
    The main function of this algorithm (variant in which clusters neighbors are marked as well)
    @param g - Graph to build spanning arbs + clusters on
    @return - list of arbs
    """
    # Compute additional graph parameters and include them as metadata
    # associated with nodes of the graph
    g = compute_additional_node_parameters(g, True)
    g.graph['k'] = edmond(g)

    # Find the primary set of spanning arborescences covering the entire graph 'G'
    GreedyArborescenceDecomposition(g)

    # Create a subgraph 'Gm' of 'G' based on the marked nodes
    Gm = return_subgraph_with_marked_nodes(g)

    # Identify and print all strongly connected components of 'Gm', such that contain at least 3 nodes
    components = list()
    components_all = list(nx.strongly_connected_components(Gm))

    for c in components_all:
        if len(c) > 2:
            components.append(list(c))

    # For each connected component, try to improve its 'k'
    # Then, find the corresponding 'k' spanning arborescences
    extra_arborescences_count = 0

    for c in components:
        Gc = nx.DiGraph(Gm.subgraph(c))

        # Try to improve the local edge connectivity by removing nodes of degree 1
        while nx.number_of_nodes(Gc) > 3:
            deg_one_nodes = return_nodes_of_degree_one(Gc)
            if len(deg_one_nodes) == 0:
                break
            Gc.remove_nodes_from(deg_one_nodes)

        if nx.number_of_nodes(Gc) < 3:
            continue

        Gc.graph['k'] = nx.edge_connectivity(Gc)
        if Gc.graph['k'] < 2:
            continue
        Gc.graph['root'] = return_gw_node_towards_the_global_root(Gc)

        # Find the set of arborescences covering the entire subgraph 'Gc'

        GreedyArborescenceDecomposition(Gc)

        # Assign the corresponding arcs of the original graph to exactly one arborescence (sometimes just 'set of links')
        # Existing assignments have priority over the new ones

        extra_arborescences = dict()

        for (u, v) in Gc.edges():
            if Gc[u][v]['attr'] == -1 or g[u][v]['attr'] > -1:
                continue
            if not Gc[u][v]['attr'] in extra_arborescences.keys():
                extra_arborescences[Gc[u][v]['attr']] = list()
            extra_arborescences[Gc[u][v]['attr']].append((u, v))

        i = 0  # Index of an extra tree

        for (arb_id, arc_list) in extra_arborescences.items():
            for (u, v) in arc_list:
                g[u][v]['attr'] = g.graph['k'] + extra_arborescences_count + i
            i += 1

        # Note that not all of the extra arborescences may have been added to 'G',
        # as the already-assigned arcs will not be included in new arborescences
        extra_arborescences_count += len(extra_arborescences)
    return get_arborescence_list(g)
