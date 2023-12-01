# Based on https://github.com/oliver306/TREE/blob/master/src/kResilienceTrees.py
# - Modified to work on directed graphs
# - Added TREE Switching
# - Added tree_choice undirected

import networkx as nx
import copy
import numpy as np


def kResilienceTrees(s: int, d: int, g: nx.Graph, version: str = "multiple", **kwargs) -> nx.DiGraph:
    """
    Build Trees on a graph
    @param s - Source node
    @param d - Destination node
    @param g - Graph
    @param version - "single" or "multiple"
    @return Graph with marked trees
    """

    # Mark graph
    g_copy = copy.deepcopy(g)
    g_copy.graph["d_incidents"] = []
    g_copy.graph["rankings"] = []
    nx.set_edge_attributes(g_copy, 0, "attr")
    nx.set_node_attributes(g_copy, 0, "attr")
    g_copy.nodes[s]["attr"] = "s"
    g_copy.nodes[s]["label"] = "s"
    g_copy.nodes[d]["attr"] = "d"
    g_copy.nodes[d]["label"] = "d"
    # Load EDPs
    try:
        edge_disjoint_paths = list(nx.edge_disjoint_paths(g_copy, s, d))
    except:
        # if an error is thrown, it means that the source node and the destination node belong
        # to different graph components; the destination node cannot be reached from the source node anymore
        return g_copy

    # sort paths long to short
    edge_disjoint_paths.sort(key=lambda x: len(x), reverse=False)
    # give numbers to paths, the higher the shorter
    no_path = 1
    for path in edge_disjoint_paths:
        for i in range(0, len(path)-1):
            if g_copy.nodes[path[i+1]]["attr"] != "d":
                g_copy.nodes[path[i+1]]["attr"] = no_path
            g_copy[path[i]][path[i+1]]["attr"] = no_path
            if g_copy.has_edge(path[i+1], path[i]):  # Directed
                g_copy[path[i+1]][path[i]]["attr"] = no_path
        no_path += 1

    d_incidents = set()

    for d_edge in g_copy.in_edges(d):
        d_incidents.add(d_edge[0])

    s_incidents = set()
    for s_edge in g_copy.edges(s):
        s_incidents.add(s_edge[1])

    if version == "multiple":
        makeMultipleTrees(g_copy, edge_disjoint_paths,
                          s, d_incidents, s_incidents, d)
    else:
        # reverse here, so we make tree from LONGEST path
        makeOneTree(g_copy, edge_disjoint_paths, reverse=True)

    # remove incident edges of d from all structures
    for d_edge in g_copy.in_edges(d):
        g_copy[d_edge[0]][d_edge[1]]['attr'] = -1
        if g_copy.has_edge(d_edge[1], d_edge[0]):
            g_copy[d_edge[1]][d_edge[0]]['attr'] = -1

    if version != "multiple":  # -> Single. postProcess after all trees are created. Multiple checks after every tree creation, so no need to check here
        postProcessTree(g_copy, s, d_incidents, edge_disjoint_paths, d)

    rankings = rankTree(g_copy, s, d, d_incidents)
    g_copy.graph["d_incidents"] = d_incidents
    g_copy.graph["rankings"] = rankings
    return g_copy


def makeOneTree(g: nx.DiGraph, edge_disjoint_paths: list[list[int]], reverse: bool = False) -> None:
    """
    Construct a tree on g and mark its edges by setting their "attr"
    @param g - Graph
    @param edge_disjoint_paths - EDPs from s to d
    @param reverse - reverse the list of EDPs
    """
    if reverse:
        edge_disjoint_paths.reverse()
        no_tree = len(edge_disjoint_paths)
    else:
        no_tree = 1
    for path in edge_disjoint_paths:
        nodes_added = 0
        for i in range(1, len(path) - 1):
            nodes = [path[i]]  # obtain a list with the nodes of the i-th path
            it = 0
            while (it < len(nodes)):
                # obtain a list with all the incident edges of nodes from the i-th path
                list_of_incident_edges = list(g.edges(nodes[it]))
                # obtain a generator of the previous edges, which provides edges with the '0' attribute
                # (meaning that they are not used by any structure yet)
                edge_candidates_gen = (edge for edge in list_of_incident_edges if
                                       g.get_edge_data(edge[0], edge[1]).get("attr") == 0)
                for edge in edge_candidates_gen:
                    if g.nodes[edge[1]]["attr"] == 0:
                        g[edge[0]][edge[1]]["attr"] = no_tree
                        if g.has_edge(edge[1], edge[0]):
                            g[edge[1]][edge[0]]["attr"] = no_tree
                        g.nodes[edge[1]]["attr"] = no_tree

                        nodes.append(edge[1])
                        nodes_added += 1

                    # we also give an attribute to the incident edges of the destination node
                    # however, tree leaves of the tree are considered to be the neighbors of the destination node
                    if g.nodes[edge[1]]["attr"] == "d":
                        g[edge[0]][edge[1]]["attr"] = no_tree
                it += 1
        no_tree = no_tree + 1 if not reverse else no_tree - 1


def makeMultipleTrees(g: nx.DiGraph, edge_disjoint_paths: list[list[int]], s: int, d_incidents: list[int], s_incidents: list[int], d: int) -> None:
    """
    Construct trees on g and mark their edges by setting their "attr"
    @param g - Graph
    @param edge_disjoint_paths - EDPs from s to d
    @param s - source node
    @param d_incidents - incidents of d (edge * -> d exists)
    @param s_incidents - incidents of s (edge s -> * exists)
    @param d - destination node
    """
    no_tree = 1
    for path in edge_disjoint_paths:
        nodes_added = 0
        for i in range(1, len(path) - 1):
            nodes = [path[i]]  # obtain a list with the nodes of the i-th path
            it = 0
            while (it < len(nodes)):
                list_of_incident_edges = list(g.edges(nodes[it]))
                edge_candidates_gen = (edge for edge in list_of_incident_edges if
                                       g.get_edge_data(edge[0], edge[1]).get("attr") == 0)

                for edge in edge_candidates_gen:
                    node_candidate_incident_attrs = [
                        g[e[0]][e[1]]["attr"] for e in g.edges(edge[1])]
                    if no_tree not in node_candidate_incident_attrs and g[edge[0]][edge[1]]["attr"] == 0 and \
                            g.nodes[edge[1]]["attr"] != "s" and g.nodes[edge[1]]["attr"] != "d":
                        g[edge[0]][edge[1]]["attr"] = no_tree
                        # For dir. graphs: Reverse edge has to have the same attr
                        if g.has_edge(edge[1], edge[0]):
                            g[edge[1]][edge[0]]["attr"] = no_tree
                        g.nodes[edge[1]]["attr"] = no_tree

                        nodes.append(edge[1])
                        nodes_added += 1

                    # we also give an attribute to the incident edges of the destination node
                    # however, tree leaves of the tree are considered to be the neighbors of the destination node
                    if g.nodes[edge[1]]["attr"] == "d":
                        g[edge[0]][edge[1]]["attr"] = no_tree

                it += 1
        postProcessTree(g, s, d_incidents,
                        edge_disjoint_paths, d, tree_attr=no_tree)
        no_tree += 1


def routeTreesSwitched(marked_g: nx.DiGraph, s: int, d: int, d_incidents: list[int], rankings: dict[int, tuple[list[int], list[int]]], unranked: bool = False, treeChoice: str = "shortest", **kwargs):
    """
    Route
    @param marked_g - graph to route on (TREES marked by either makeOneTree() or makeMultipleTrees())
    @param s - source node
    @param d - destination node
    @param d_incidents - List of neighbors of destination
    @param rankings - returned by rankTree()
    @param unranked - false -> use rankDfs() to determine tree traversal order
    @param treeChoice - "shortest"/"undirected"
    """
    fails = []
    for edge in marked_g.edges:
        data = marked_g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))
    hops = 0
    if s in d_incidents and (s, d) not in fails:
        return True, 1, [(s, d)], {'ReverseEdgesUsed': 0}

    trees_attributes = getTreeOrder(
        rankings, marked_g, treeChoice=treeChoice, d_incidents=d_incidents)
    routed_paths = []
    found = False
    curr_node = s
    reverse_edges = 0

    def try_path(attr: int):
        nonlocal curr_node, hops, found, reverse_edges
        """Route on a single tree, given by attr"""
        T = nx.Graph()  # we reconstruct the tree
        for node1, node2, data in marked_g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)
        if s not in list(T.nodes) or curr_node not in list(T.nodes):
            return  # Switch to next tree
        # In directed case: s-a-b-*-d guarantees actual existence of edges s->a->b->*->d
        # Existence of reverse edge (i.e. s<-a, a<-b, ...) needs to be checked for every edge!
        for node1, node2, data in marked_g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)

        dfs_edge_order_list = [(n1, n2, label) for n1, n2, label in nx.dfs_labeled_edges(
            T, s) if label != "nontree" and n1 != n2]
        if not unranked and attr in rankings:
            dfs_edge_order_list = rankDfs(
                T, s, d, dfs_edge_order_list, rankings[attr])

        if curr_node != s:
            # Reorder dfs to start at curr_node, not s
            new_dfs = []
            visited = []
            dfs_node = curr_node
            while dfs_node != s:
                # Add all edges following dfs_node
                for (u, v, l) in dfs_edge_order_list:
                    if l == "forward" and u == dfs_node and v not in visited:
                        new_dfs += dfs_edge_order_list[dfs_edge_order_list.index(
                            (u, v, l)):dfs_edge_order_list.index((u, v, "reverse"))+1]
                # Proceed to previous node until we reach s
                new_dfs += [(u, v, l) for (u, v, l) in dfs_edge_order_list if v ==
                            dfs_node and l == "reverse"]  # Adds one edge
                visited.append(dfs_node)
                dfs_node = new_dfs[-1][0]

            dfs_edge_order_list = new_dfs
        final_dfs = removeFails_dir(dfs_edge_order_list, fails)
        for n1, n2, label in final_dfs:
            if not marked_g.has_edge(n1, n2) or (n1, n2) in fails:  # reverse edge failed
                return  # Switch to next tree
            routed_paths.append((n1, n2))
            curr_node = n2
            hops += 1
            if label == "reverse":
                reverse_edges += 1
            if n2 in d_incidents and (n2, d) not in fails:
                routed_paths.append((n2, d))
                curr_node = d
                hops += 1
                found = True
                break

    for attr in trees_attributes:
        if found:
            break
        try_path(attr)
    return found, hops, routed_paths, {'ReverseEdgesUsed': reverse_edges}


def routeTrees(g: nx.Graph, s: int, d: int, d_incidents: list[int], rankings: dict[int, tuple[list[int], list[int]]], unranked: bool = False, treeChoice: str = "shortest"):
    """
    Route
    @param g - graph to route on (TREES marked by either makeOneTree() or makeMultipleTrees())
    @param s - source node
    @param d - destination node
    @param d_incidents - List of neighbors of destination
    @param rankings - returned by rankTree()
    @param unranked - false -> use rankDfs() to determine tree traversal order
    @param treeChoice - "shortest"/"undirected"
    """
    fails = []
    for edge in g.edges:
        data = g.get_edge_data(*edge)
        if data['failed']:
            fails.append((edge[0], edge[1]))
    hops = 0
    reverse_edges = 0
    if s in d_incidents and (s, d) not in fails:
        return True, 1, [(s, d)], {'ReverseEdgesUsed': 0}

    trees_attributes = getTreeOrder(
        rankings, g, treeChoice=treeChoice, d_incidents=d_incidents)
    routed_paths = []
    found = False
    for attr in trees_attributes:
        if found:
            break
        T = nx.Graph()  # we reconstruct the tree
        # In directed case: s-a-b-*-d guarantees actual existence of edges s->a->b->*->d
        # Existence of reverse edge (i.e. s<-a, a<-b, ...) needs to be checked for every edge!
        for node1, node2, data in g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)

        if s not in list(T.nodes):
            continue

        dfs_edge_order_list = [(n1, n2, label) for n1, n2, label in nx.dfs_labeled_edges(
            T, s) if label != "nontree" and n1 != n2]
        if not unranked and attr in rankings:
            dfs_edge_order_list = rankDfs(
                T, s, d, dfs_edge_order_list, rankings[attr])
        final_dfs = removeFails_dir(dfs_edge_order_list, fails)
        for n1, n2, label in final_dfs:
            # reverse edge failed, packet lost
            if not g.has_edge(n1, n2) or (n1, n2) in fails:
                return False, hops, routed_paths, {'ReverseEdgesUsed': reverse_edges}
            routed_paths.append((n1, n2))
            hops += 1
            if label == "reverse":
                reverse_edges += 1
            if n2 in d_incidents and (n2, d) not in fails:
                routed_paths.append((n2, d))
                hops += 1
                found = True
                break

    return found, hops, routed_paths, {'ReverseEdgesUsed': reverse_edges}


def postProcessTree(g: nx.DiGraph, s: int, d_incidents: list[int], edge_disjoint_paths: list[list[int]], d: int, tree_attr: list[int] | None = None):
    """
    Prune created tree branches that do not lead to d
    @param g - graph with trees marked
    @param s - source node
    @param d_incidents - List of neighbors of d
    @param edge_disjoint_paths - EDPs from s to d
    @param d - destination node
    @param tree_attr - Attrs of trees to postProcess
    """
    # we will test if we can still reach the destination node; for this we will analyze each subgraph (tree) until we will find the destination node
    trees_attributes = []
    if tree_attr is None:
        for node1, node2, data in g.edges(data=True):
            if data['attr'] not in trees_attributes:
                if int(data['attr']) > 0:
                    trees_attributes.append(data['attr'])
    else:
        trees_attributes.append(tree_attr)

    for attr in trees_attributes:
        all_edges = []
        T = nx.Graph()  # we reconstruct the tree
        for node1, node2, data in g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)
                all_edges.append((node1, node2))
        if s in list(T.nodes):
            dfs_edge_order_list = [(n1, n2, label) for n1, n2, label in dfs_labeled_edges_target(
                T, s, d) if label != "nontree" and n1 != n2]
            dfs_edges = [(n1, n2) for n1, n2, label in dfs_edge_order_list]
            # Remove paths relying on non-existent forward edges.
            if g.is_directed():
                delete_until = ()
                for n1, n2, label in dfs_edge_order_list.copy():
                    if delete_until != ():
                        if n1 == delete_until[0] and n2 == delete_until[1]:
                            dfs_edge_order_list.remove((n1, n2, label))
                            delete_until = ()
                        else:
                            dfs_edge_order_list.remove((n1, n2, label))
                            if g.has_edge(n1, n2):
                                g[n1][n2]["attr"] = 0
                            if g.has_edge(n2, n1):
                                g[n2][n1]["attr"] = 0
                    elif label == "forward":
                        if not g.has_edge(n1, n2):
                            dfs_edge_order_list.remove((n1, n2, label))
                            if g.has_edge(n1, n2):
                                g[n1][n2]["attr"] = 0
                            if g.has_edge(n2, n1):
                                g[n2][n1]["attr"] = 0
                            delete_until = (n1, n2)
            for n1, n2 in all_edges:  # Nontree -> some edges are not checked again, set them to 0
                if not ((n1, n2) in dfs_edges or (n2, n1) in dfs_edges):
                    if g.has_edge(n1, n2):
                        g[n1][n2]["attr"] = 0
                    if g.has_edge(n2, n1):
                        g[n2][n1]["attr"] = 0
            good_branch_nodes = set()
            visited_nodes = set()
            visited_nodes.add(dfs_edge_order_list[0][0])
            delete_mode = False
            for i in range(len(dfs_edge_order_list)):
                n1, n2, label = dfs_edge_order_list[i]
                if label == "forward":
                    visited_nodes.add(n2)
                elif label == "reverse":
                    visited_nodes.remove(n2)

                if label == "forward" or n2 in good_branch_nodes:
                    delete_mode = False

                if delete_mode:
                    if g.nodes[n2]["attr"] != "d":  # Don't overwrite destination
                        g.nodes[n2]["attr"] = 0
                    if g.has_edge(n1, n2):
                        g[n1][n2]["attr"] = 0
                    if g.has_edge(n2, n1):
                        g[n2][n1]["attr"] = 0

                if i < len(dfs_edge_order_list) - 1:
                    n1_next, n2_next, label_next = dfs_edge_order_list[i+1]
                    if label == "forward" and label_next == "reverse" and n2 not in d_incidents:
                        delete_mode = True
                    elif n2 in d_incidents:
                        [good_branch_nodes.add(el) for el in visited_nodes]
                        if edge_disjoint_paths is not None:
                            cnt = 0
                            for path in edge_disjoint_paths:
                                if n2 not in path:
                                    cnt += 1
                                else:
                                    break


def getTreeOrder(rankings: dict[int, tuple[list[int], list[int]]], g: nx.DiGraph, treeChoice: str = "shortest", d_incidents: list[int] = []) -> list[int]:
    """
    @param rankings - returned by rankTree()
    @param g - graph with trees marked
    @param treeChoice - "shortest"/"undirected"
    @param d_incidents - List of neighbors of d
    @return - List of trees attrs in order
    """
    order = []
    if d_incidents != []:
        for key in rankings:
            for di in d_incidents:
                if di in rankings[key][0]:
                    rankings[key][0][di] += [1]  # Incident has distance 1
    if treeChoice != "undirected":
        for key in rankings:
            if len(rankings[key][0]) == 0:
                order.append((key, 1))
            else:
                if treeChoice == "average":
                    tmpval = np.mean(list(rankings[key][0].values())[0])
                elif treeChoice == "edgeCount":
                    tmpval = len(list(rankings[key][0].values()))
                else:  # treeChoice == "shortest"
                    tmpval = np.min(list(rankings[key][0].values())[0])
                order.append((key, tmpval))
    else:  # undirected
        dir_edges = {}
        for node1, node2, data in g.edges(data=True):
            # Edge directed
            if data['attr'] != 0 and data['attr'] != -1 and not g.has_edge(node2, node1):
                if data['attr'] in dir_edges:
                    dir_edges[data['attr']] += 1
                else:
                    dir_edges[data['attr']] = 1
        for key in rankings:
            if len(rankings[key][0]) == 0:
                tmpval = 1
            else:
                tmpval = np.min(list(rankings[key][0].values())[0])
            if key in dir_edges:
                # Sorted primarily by length, then number of dir. edges
                order.append((key, (tmpval, dir_edges[key])))
                # order.append((key, (tmpval, dir_edges[key])))  # Sorted primarily by length, then number of dir. edges
            else:
                # Only undirected edges -> traverse before any other
                order.append((key, (0, -1)))
    order.sort(key=lambda x: x[1])
    final_order = [el[0] for el in order]
    return final_order


def rankTree(g: nx.Graph, s: int, d: int, d_incidents: list[int], trees_changed: list[int] = None) -> dict[int, tuple[list[int], list[int]]]:
    """
    @param g - Graph with trees to rank
    @param s - source node
    @param d - destination node
    @param d_incidents - List of neighbors of d
    @param trees_changed - List of attrs of trees to rank
    @return dict: attr -> ([distances], [directions])
    """
    # Directed graphs are handled as if every edge was undirected
    # if trees_changed not given, just take all attrs in graph (even the one that are not trees, but still paths)
    if trees_changed is None:
        trees_changed = []
        for node1, node2, data in g.edges(data=True):
            if data['attr'] not in trees_changed and int(data['attr']) > 0:
                trees_changed.append(data['attr'])

    trees_ranked = {}
    for attr in trees_changed:
        T = nx.Graph()  # we reconstruct the tree
        for node1, node2, data in g.edges(data=True):
            if data is not None and 'attr' in data and data['attr'] == attr:
                T.add_edge(node1, node2)
        if s not in list(T.nodes):
            continue

        dfs_edge_order_list = [(n1, n2, label) for n1, n2, label in nx.dfs_labeled_edges(
            T, s) if label != "nontree" and n1 != n2]

        branching_dict = {}
        direction_dict = {}
        best_dict = {}
        travelled_nodes = set()
        for i in range(len(dfs_edge_order_list)):
            n1, n2, label = dfs_edge_order_list[i]
            # if we can reach d from here and its the best route so far, lock temporary distances
            if label == "forward" and n2 in d_incidents:
                for key in branching_dict:
                    if key not in best_dict:
                        best_dict[key] = []
                        direction_dict[key] = []

                    if key in travelled_nodes:
                        # plus one bc we need one more hop to d
                        best_dict[key].append(branching_dict[key] + 1)
                        direction_dict[key].append(n2)

            # add current node as travelled
            travelled_nodes.add(n2)

            # remove current node from travelled if we move backwards from a node already visited
            if label == "reverse" and n2 in travelled_nodes:
                travelled_nodes.remove(n2)

            if label == "forward":
                for node in travelled_nodes:
                    if node in branching_dict:
                        branching_dict[node] += 1
                    else:
                        branching_dict[node] = 1

            if label == "reverse":
                for node in travelled_nodes:
                    branching_dict[node] -= 1

        trees_ranked[attr] = (best_dict, direction_dict)
    return trees_ranked


def rankDfs(T: nx.Graph, s: int, d: int, dfs: list[tuple[int, int, str]], ranking: tuple[list[int], list[int]]) -> list[tuple[int, int, str]]:
    """
    prepare final depth first traversal according to ranking (fastest route to d first)
    @param T - Tree
    @param s - Source node
    @param d - Destination node
    @param dfs - Depth-first search 
    @param ranking - returned by rankTree()[attr]
    @return ordered traversal
    """
    successor_dict = nx.dfs_successors(T, s)
    ranking_processed = [{}, {}]

    def getAllSuccessors(node, succs, set):
        if node in succs:
            for succ in succs[node]:
                set.add(succ)
                getAllSuccessors(succ, succs, set)

    for node in ranking[0]:
        if node not in successor_dict:
            continue

        # write direct neighbor to ranking dict instead of very last node on path -> enable local routing
        neighs = successor_dict[node]
        for neigh in neighs:
            all_succs = set()
            getAllSuccessors(neigh, successor_dict, all_succs)
            for succ in all_succs:
                if succ in ranking[1][node]:
                    ranking[1][node][ranking[1][node].index(succ)] = neigh

        # if multiple paths go over same direct neighbor, take shortest one
        ranking_processed[0][node] = []
        ranking_processed[1][node] = []

        for distinct in np.unique(ranking[1][node]):
            indices = [i for i, x in enumerate(
                ranking[1][node]) if x == distinct]

            tmpmin = np.min([ranking[0][node][idx] for idx in indices])
            ranking_processed[0][node].append(tmpmin)
            ranking_processed[1][node].append(distinct)

        # sort neighbor rankings so that shortest route is first
        hops = ranking_processed[0][node].copy()
        hops_sorted = sorted(hops)
        dirs_sorted = []
        for i in range(len(hops_sorted)):
            dirs_sorted.append(
                ranking_processed[1][node][hops.index(hops_sorted[i])])
            hops[hops.index(hops_sorted[i])] = -1

        ranking_processed[0][node] = hops_sorted
        ranking_processed[1][node] = dirs_sorted

    for key in ranking_processed[0]:
        i = 0
        swaps_idx = [-1] * len(ranking_processed[0][key])
        for n1, n2, label in dfs:
            if n1 == key and label == "forward":
                swaps_idx[ranking_processed[1][key].index(
                    n2)] = i
            if n1 == key and label == "reverse":
                swaps_idx[ranking_processed[1][key].index(n2)] = (
                    swaps_idx[ranking_processed[1][key].index(n2)], i + 1)

            i += 1
        # Directed -> shortened dfs
        swaps_idx = [x for x in swaps_idx if x != -1]
        if len(swaps_idx) == 0:  # Nothing to reorder
            continue
        range_start = min(swaps_idx, key=lambda x: x[0])[0]
        range_end = max(swaps_idx, key=lambda x: x[1])[1]
        # dfs_reordered = [None] * len(dfs)
        dfs_reordered = []
        for si in swaps_idx:
            dfs_reordered.extend(dfs[si[0]: si[1]])

        dfs[range_start: range_end] = dfs_reordered

    return dfs


def removeFails_dir(dfs: list[tuple[int, int, str]], fails: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """
    Check the dfs: remove failed forward edges, stop after encountering failed reverse edge
    @param dfs - dfs to remove fails from
    @param fails - list of failed edges
    """
    for fail in fails:
        idx = 0
        start_idx = -1
        end_idx = -1
        search_mode = False
        failed_edge = (-1, -1)
        for n1, n2, label in dfs:
            if n1 == fail[0] and n2 == fail[1] and label == "forward":
                search_mode = True
                failed_edge = fail
                start_idx = idx

            if search_mode and n1 == failed_edge[0] and n2 == failed_edge[1] and label == "reverse":
                end_idx = idx
                break  # new

            idx += 1
        if start_idx > -1 and end_idx > -1:
            for i in range(end_idx, start_idx-1, -1):
                del dfs[i]
    for i, (u, v, l) in enumerate(dfs):
        if l == "reverse":
            dfs[i] = (v, u, l)
    return dfs


def dfs_labeled_edges_target(G, source=None, target=None, depth_limit=None):
    """Based on nx.dfs_labeled_edges. Added target: Upon reaching the target, we turn around.
    """
    # Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    # by D. Eppstein, July 2004.
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        yield start, start, "forward"
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child in visited:
                    yield parent, child, "nontree"
                else:
                    yield parent, child, "forward"
                    visited.add(child)
                    if depth_now > 1 and child != target:
                        stack.append((child, depth_now - 1, iter(G[child])))
                    else:
                        yield parent, child, "reverse"
            except StopIteration:
                stack.pop()
                if stack:
                    yield stack[-1][0], parent, "reverse"
        yield start, start, "reverse"


def build_routes(g: nx.Graph, s: int, d: int, **kwargs):
    """
    @param g - Graph
    @param s - Source node
    @param d - Destination node
    """
    return kResilienceTrees(s, d, g, **kwargs)


def route(marked_g: nx.Graph, s: int, d: int, switching, treeChoice, **kwargs):
    """
    @param marked_g - Graph
    @param s - Source node
    @param d - Destination node
    @param switching - Tree switching
    @param treeChoice - "shortest"/"undirected"
    """
    d_incidents = marked_g.graph["d_incidents"]
    rankings = marked_g.graph["rankings"]
    if switching:
        return routeTreesSwitched(marked_g.copy(), s, d, d_incidents, rankings, treeChoice=treeChoice, **kwargs)
    else:
        return routeTrees(marked_g.copy(), s, d, d_incidents, rankings, treeChoice=treeChoice)
