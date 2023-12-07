# Based on https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/blob/master/extra_links.py
# - Modified to work on directed graphs (DFS-Traversal, routing assumes no existence of reverse edge)

import sys
import networkx as nx
from RoutingAlgos.BonsaiImpl import *


def KeepForwardingPrecomputation(g: nx.Graph, version=2) -> None:
    """
    Precomputations for Keep Forwarding Routing. Precomputation is saved in g.graph["precomp"]
    @param g: Graph to build routes on
    """
    nx.set_edge_attributes(g, False, "inter_component_link")
    nx.set_edge_attributes(g, "", "link_type")
    # Normal a links: default priority.
    # non-tree links and intercomponent links are handled separately with low/high prio
    nx.set_edge_attributes(g, False, "normal_a_link")
    nx.set_edge_attributes(g, 1, "attr")  # So every edge gets drawn as used
    d = g.graph['root']
    n = len(g.nodes())
    dist = nx.shortest_path_length(g, target=d)
    edge_weight = {}
    node_weight = {v: 0 for v in g.nodes()}
    dist_nodes = {i: set() for i in range(n)}  # nodes with distance i
    dist_nodes[sys.maxsize] = set()  # nodes that can't reach destination
    down_links = {v: set() for v in g.nodes()}
    A_links = {v: set() for v in g.nodes()}
    a_link_low_prio = {v: set() for v in g.nodes()}  # node -> {nodes}
    a_link_high_prio = {v: set() for v in g.nodes()}  # node -> {nodes}
    up_links = {v: set() for v in g.nodes()}
    # Annotate with distances
    for (u, v) in g.edges():
        if not u in dist:
            dist[u] = sys.maxsize
        if not v in dist:
            dist[v] = sys.maxsize
        if u == v:
            continue
        if dist[u] > dist[v]:
            edge_weight[(u, v)] = n*n
            down_links[u].add(v)
            g[u][v]["link_type"] = "down-link"
        elif dist[u] == dist[v]:
            edge_weight[(u, v)] = n
            A_links[u].add(v)
            g[u][v]["link_type"] = "a-link"
        elif dist[u] < dist[v]:
            if dist[v] != sys.maxsize:
                edge_weight[(u, v)] = 1
                up_links[u].add(v)
                g[u][v]["link_type"] = "up-link"
            else:
                edge_weight[(u, v)] = 0
                g[u][v]["link_type"] = "unused"
        node_weight[u] += edge_weight[(u, v)]
        dist_nodes[dist[u]].add(u)
        dist_nodes[dist[v]].add(v)
    label = {}  # eulerian label
    label_size = {}  # eulerian size
    for k, v in dist_nodes.items():  # distance, all nodes with equal distance
        if len(v) < 2:
            continue
        subgraph = g.subgraph(v)
        components = list(nx.strongly_connected_components(subgraph))
        node_component = {}  # node -> component index
        for i, component in enumerate(components):
            for node in component:
                node_component[node] = i
            count = 0
            try:
                for (u, v) in nx.eulerian_circuit(g.subgraph(component)):
                    if not g.has_edge(u, v):
                        continue
                    g[u][v]["normal_a_link"] = True
                    label[(u, v)] = count
                    count += 1
                    label_size[(u, v)] = g.subgraph(
                        component).number_of_edges()
            except:  # No eulerian circuit available. Should only happen on directed graphs
                # DFS
                dfs_edges = nx.dfs_labeled_edges(g.subgraph(component))
                dfs_edges = [(u, v, l) if l != "reverse" else (v, u, l)
                             for (u, v, l) in dfs_edges]
                tree_edges = [(u, v) for (u, v, l) in dfs_edges if l !=
                              'nontree' and u != v and g.has_edge(u, v)]
                nontree_edges = [(u, v) for (u, v) in g.subgraph(
                    component).edges() if (u, v) not in tree_edges]
                for (u, v) in tree_edges:
                    g[u][v]["normal_a_link"] = True
                    label[(u, v)] = count
                    count += 1
                for (u, v) in tree_edges:
                    label_size[(u, v)] = len(tree_edges)
                for (u, v) in nontree_edges:
                    a_link_low_prio[u].add(v)
        for (u, v) in subgraph.edges():
            if node_component[u] != node_component[v]:  # Unused edge between components
                g[u][v]["inter_component_link"] = True
                u_nodes = components[node_component[u]]
                v_nodes = components[node_component[v]]
                if len(u_nodes) > len(v_nodes):  # Low prio add
                    a_link_low_prio[u].add(v)
                else:  # High prio add
                    a_link_high_prio[u].add(v)
    A_links = {key: {v for v in val if g[key][v]["normal_a_link"]}
               for (key, val) in A_links.items()}
    if version == 1:  # Reset links between components for v1
        a_link_low_prio = {v: set() for v in g.nodes()}  # node -> {nodes}
        a_link_high_prio = {v: set() for v in g.nodes()}  # node -> {nodes}
    g.graph["precomp"] = [label_size, label, node_weight, down_links,
                          A_links, up_links, a_link_low_prio, a_link_high_prio]


def KeepForwardingRouting(s: int, d: int, g: nx.Graph) -> tuple[bool, int, list[tuple[int, int]]]:
    """
    Route
    @param s - source node
    @param d - destination node
    @param g - graph to route on
    @return (found, hops, route (as edge list))
    """

    def inc_a_count(a_count, label_size, incoming_link, A_links, s):
        a_count = (a_count+1) % label_size[incoming_link]
        while len([i for i in A_links if label[(s, i)] == a_count]) == 0:
            a_count = (a_count+1) % label_size[incoming_link]
        return a_count

    fails = []
    for edge in g.edges:
        data = g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))

    [label_size, label, node_weight, down_links, A_links, up_links,
        a_link_low_prio, a_link_high_prio] = g.graph["precomp"]

    for (u, v) in fails:
        down_links[u] -= {v}
        A_links[u] -= {v}
        up_links[u] -= {v}
        a_link_low_prio[u] -= {v}
        a_link_high_prio[u] -= {v}
    g.remove_edges_from(fails)
    hops = 0

    # add edges taken to this list when the first failure has been encountered...
    detour_edges = []
    n = len(g.nodes())
    incoming_link = (s, s)
    incoming_node = s
    while (s != d):
        # remove incoming node from all link lists
        curr_dl = list(down_links[s])
        if incoming_node in curr_dl:
            curr_dl.remove(incoming_node)
        curr_al = list(A_links[s])
        if incoming_node in curr_al:
            curr_al.remove(incoming_node)
        curr_ul = list(up_links[s])
        if incoming_node in curr_ul:
            curr_ul.remove(incoming_node)

        # sort up/down according to weights (higher->earlier) and a-list according to labels (lower->earlier)
        curr_dl = sorted(curr_dl, key=lambda x: int(
            node_weight[x]), reverse=True)
        curr_al = sorted(curr_al, key=lambda x: int(
            label[(s, x)]), reverse=False)
        curr_ul = sorted(curr_ul, key=lambda x: int(
            node_weight[x]), reverse=True)

        # init for a links
        a_count = -1  # init counter for label of link (for safety)

        # if incoming was a-link, get correct counter
        if s in list(A_links[incoming_node]):
            a_count = label[incoming_link]  # get label from incoming link
        elif len(curr_al) > 0:
            a_count = label[(s, curr_al[0])]

        if len(curr_dl) > 0:  # if down list is not empty, set nxt as first element of down list
            curr_list = curr_dl
            nxt = curr_list[0]
        else:
            if len(a_link_high_prio[s]) > 0:  # high prio a-link
                nxt = list(a_link_high_prio[s])[0]
            else:
                if len(curr_al) > 0:  # if a list is not empty, set nxt as next a link: if incoming is a-link, then next, else, as first element of down list
                    curr_list = curr_al
                    # if incoming was a link
                    if s in list(A_links[incoming_node]):
                        # increase counter by 1
                        a_count = inc_a_count(
                            a_count, label_size, incoming_link, curr_al, s)
                        nxt = next(
                            i for i in curr_al if label[(s, i)] == a_count)
                    else:  # if incoming was not a-link
                        nxt = curr_list[0]
                        a_count = label[(s, curr_list[0])]
                else:
                    if len(a_link_low_prio[s]) > 0:
                        nxt = list(a_link_low_prio[s])[0]
                    # if a list is not empty, set nxt as first element of down list
                    elif len(curr_ul) > 0:
                        curr_list = curr_ul
                        nxt = curr_list[0]
                    else:  # note: this should not happen, as we did not yet check if the next link is failed, but added for good measure...
                        if g.has_edge(s, incoming_node) and not (s, incoming_node) in fails:
                            nxt = incoming_node
                        else:
                            nxt = s

        if s == nxt:  # Self loop: Found no valid edge to send the packet
            return (False, hops, detour_edges)
        detour_edges.append((s, nxt))
        hops += 1
        n_end = n*n+20
        if hops > n_end:  # n*n*n:  #to kill early, later set back to n*n*n
            # probably a loop, return
            return (False, hops, detour_edges)
        incoming_link = (s, nxt)
        incoming_node = s
        s = nxt
    return (True, hops, detour_edges)
