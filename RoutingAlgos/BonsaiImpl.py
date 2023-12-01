# Based on https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/blob/master/arborescences.py
# - Modified to work on directed graphs (edmond to calculate number of arbs)


# import igraph as ig
import networkx as nx
from heapq import heappush, heappop
import random
import sys


def edmond(g: nx.DiGraph) -> int:
    """
    Based on edmonds branching theorem.
    @param g - A graph, may be directed
    @return - Number of possible edge-disjoint, spanning arbs in a graph
    """
    root = g.graph['root']
    minimum = sys.maxsize
    for node in g.nodes:
        if node == root:
            continue
        try:
            n_paths = len(list(nx.edge_disjoint_paths(g, node, root)))
        except:
            return 0
        minimum = min(minimum, n_paths)
    return minimum


swappy = []


def reset_arb_attribute(g: nx.Graph) -> None:
    """
    Reset the arb attribute for all edges to -1, i.e., no arborescence assigned yet
    """
    for (u, v) in g.edges():
        g[u][v]['attr'] = -1


def swap(g: nx.Graph, u1: int, v1: int, u2: int, v2: int) -> None:
    """
    Given a graph g and edge (u1, v1) and edge (u2,v2) swap the arborescences they belong to (will crash if the edges dont belong to the graph)
    """
    i1 = g[u1][v1]['attr']
    i2 = g[u2][v2]['attr']
    g[u1][v1]['attr'] = i2
    g[u2][v2]['attr'] = i1


def get_arborescence_dict(g: nx.DiGraph) -> dict[int, list[nx.DiGraph]]:
    """
    Given a graph return the arborescences in a dictionary with indices as keys
    @param g - Graph with marked arborescences
    @return dict attr -> [arbs]
    """
    arbs = {}
    for (u, v) in g.edges():
        index = g[u][v]['attr']
        if index not in arbs:
            arbs[index] = nx.DiGraph()
            arbs[index].graph['root'] = g.graph['root']
            arbs[index].graph['index'] = index
        arbs[index].add_edge(u, v)
    return arbs


def get_arborescence_list(g: nx.DiGraph) -> list[nx.DiGraph]:
    """
    @param g - Graph with marked arborescences
    @return - list of all arborescences in the graph
    """
    arbs = get_arborescence_dict(g)
    sorted_indices = sorted([i for i in arbs.keys() if i >= 0])
    return [arbs[i] for i in sorted_indices]


def TestCut(g: nx.Graph, s: int, d: int) -> int:
    """
    return the edge connectivity of g between s and t
    """
    '''
    G_ig = ig.Graph(directed=g.is_directed())
    G_ig.add_vertices(g.nodes)  # KEYERROR HERE
    G_ig.add_edges(g.edges())
    c = G_ig.edge_connectivity(s, d)
    return c'''
    return nx.edge_connectivity(g, s, d)


def FindRandomTree(g: nx.Graph, k: int) -> nx.DiGraph():
    """
    @return a random arborescence rooted at the root
    """
    T = nx.DiGraph()
    T.add_node(g.graph['root'])
    R = {g.graph['root']}
    dist = dict()
    dist[g.graph['root']] = 0
    # heap of all border edges in form [(edge metric, (e[0], e[1])),...]
    hi = []
    preds = sorted(g.predecessors(
        g.graph['root']), key=lambda k: random.random())
    for x in preds:
        hi.append((0, (x, g.graph['root'])))
        if k > 1:
            continue
    while len(hi) > 0:  # len(h) > 0:
        (d, e) = random.choice(hi)
        hi.remove((d, e))
        g.remove_edge(*e)
        if e[0] not in R and (k == 1 or TestCut(g, e[0], g.graph['root']) >= k-1):
            dist[e[0]] = d+1
            R.add(e[0])
            preds = sorted(g.predecessors(e[0]), key=lambda k: random.random())
            for x in preds:
                if x not in R:
                    hi.append((d+1, (x, e[0])))
            T.add_edge(*e)
        else:
            g.add_edge(*e)
    if len(R) < len(g.nodes()):
        pass
        # print("Couldn't find next edge for tree with root %s" % str(r))
        # sys.stdout.flush()
    return T


def FindTree(g: nx.Graph, k: int) -> nx.DiGraph:
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
        if k > 1:
            continue
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
        # sys.stdout.flush()
        pass
    return T


def RandomTrees(g):
    """
    associate random trees as arborescences with g
    """
    reset_arb_attribute(g)
    gg = g.to_directed()
    # K = g.graph['k']
    k = edmond(gg)
    K = k
    while k > 0:
        T = FindRandomTree(gg, k)
        if T is None:
            return None
        for (u, v) in T.edges():
            g[u][v]['attr'] = K-k
        gg.remove_edges_from(T.edges())
        k = k-1
    return K


def GreedyArborescenceDecomposition(g):
    """
    associate a greedy arborescence decomposition with g
    """
    reset_arb_attribute(g)
    gg = g.to_directed()
    # K = g.graph['k']
    k = edmond(gg)
    K = k
    while k > 0:
        T = FindTree(gg, k)
        if T is None:
            return None
        for (u, v) in T.edges():
            g[u][v]['attr'] = K-k
        gg.remove_edges_from(T.edges())
        k = k-1
    return K


# Helper class (some algorithms work with Network, others without),
# methods as above
class Network:
    # initialize variables
    def __init__(self, g, K, root):
        self.g = g
        self.K = K
        self.root = root
        self.arbs = {}
        self.build_arbs()
        self.dist = nx.shortest_path_length(self.g, target=root)

    # create arbs data structure from edge attributes
    def build_arbs(self):
        self.arbs = {index: nx.DiGraph() for index in range(self.K)}
        for (u, v) in self.g.edges():
            index = self.g[u][v]['attr']
            if index > -1:
                self.arbs[index].add_edge(u, v)

    # create arborescence for index given edge attributes
    def build_arb(self, index):
        self.arbs[index] = nx.DiGraph()
        for (u, v) in self.g.edges():
            if self.g[u][v]['attr'] == index:
                self.arbs[index].add_edge(u, v)

    # return graph of edges not assigned to any arborescence
    def rest_graph(self, index):
        rest = nx.DiGraph()
        for (u, v) in self.g.edges():
            i = self.g[u][v]['attr']
            if i > index or i == -1:
                rest.add_edge(u, v)
        return rest

    # add edge (u,v) to arborescence of given index
    def add_to_index(self, u, v, index):
        old_index = self.g[u][v]['attr']
        self.g[u][v]['attr'] = index
        if index > -1:
            self.arbs[index].add_edge(u, v)
        if old_index > -1:
            self.build_arb(old_index)

    # remove edge (u,v) from the arborescence it belonged to
    def remove_from_arbs(self, u, v):
        old_index = self.g[u][v]['attr']
        self.g[u][v]['attr'] = -1
        if old_index > -1:
            self.build_arb(old_index)

    # swap arborescence assignment for edges (u1,v1) and (u2,v2)
    def swap(self, u1, v1, u2, v2):
        i1 = self.g[u1][v1]['attr']
        i2 = self.g[u2][v2]['attr']
        self.g[u1][v1]['attr'] = i2
        self.g[u2][v2]['attr'] = i1
        self.build_arb(i1)
        self.build_arb(i2)

    # return true iff graph with arborescence index i is a DAG
    def acyclic_index(self, i):
        return nx.is_directed_acyclic_graph(self.arbs[i])

    # return true if graoh of given index is really an arborescence
    def is_arb(self, index):
        arb = self.arbs[index]
        root = self.root
        if root in arb.nodes():
            distA = nx.shortest_path_length(arb, target=root)
        else:
            return False
        for v in arb.nodes():
            if v == root:
                continue
            if arb.out_degree(v) != 1 or v not in distA:
                return False
            # if self.K - index > 1:
            #   rest = self.rest_graph(index)
               # if not v in rest.nodes() or TestCut(self.rest_graph(index), v, root) < self.K-index-1:
                #    return False
        return True

    # return nodes that are part of arborescence for given index
    def nodes_index(self, index):
        if index > -1:
            arb = self.arbs[index]
            l = list(arb.nodes())
            for u in l:
                if u != self.root and arb.out_degree(u) < 1:
                    arb.remove_node(u)
            return arb.nodes()
        else:
            return self.g.nodes()

    # return number of nodes in all arborescences
    def num_complete_nodes(self):
        return len(self.complete_nodes())

    # return nodes which belong to all arborescences
    def complete_nodes(self):
        c = set(self.g.nodes())
        for arb in self.arbs.values():
            c = c.intersection(set(arb.nodes()))
        return c

    # return number of nodes in all arborescences
    def shortest_path_length(self, index, u, v):
        return nx.shortest_path_length(self.arbs[index], u, v)

    # return true iff node v is in shortest path from node u to root in
    # arborescence of given index
    def in_shortest_path_to_root(self, v, index, u):
        return (v in nx.shortest_path(self.arbs[index], u, self.root))

    # return predecessors of node v in g (as a directed graph)
    def predecessors(self, v):
        return self.g.predecessors(v)


def prepareDS(n: Network, h, dist, reset=True):
    """
    set up network data structures before using them
    """
    if reset:
        reset_arb_attribute(n.g)
    for i in range(n.K):
        dist.append({n.root: 0})
        preds = sorted(n.g.predecessors(n.root), key=lambda k: random.random())
        heapT = []
        for x in preds:
            heappush(heapT, (0, (x, n.root)))
        h.append(heapT)
        n.arbs[i].add_node(n.root)


def trySwap(n, h, index):
    """
    try to swap an edge on arborescence index for network with heap h
    """
    ni = list(n.nodes_index(index))
    for v1 in ni:
        for u in n.g.predecessors(v1):
            index1 = n.g[u][v1]['attr']
            if u == n.root or index1 == -1 or u in ni:
                continue
            for v in n.g.successors(u):
                if n.g[u][v]['attr'] != -1 or v not in n.nodes_index(index1):
                    continue
                if not n.in_shortest_path_to_root(v1, index1, v):
                    n.add_to_index(u, v, index)
                    n.swap(u, v, u, v1)
                    if n.is_arb(index) and n.is_arb(index1):
                        update_heap(n, h, index)
                        update_heap(n, h, index1)
                        add_neighbors_heap(n, h, [u, v, v1])
                        return True
                    # print("undo swap")
                    n.swap(u, v, u, v1)
                    n.remove_from_arbs(u, v)
    return False


def update_heap(n, h, index):
    """
    add a new item to the heap
    """
    new = []
    for (d, e) in list(h[index]):
        if e[1] in n.arbs[index].nodes():
            d = n.shortest_path_length(index, e[1], n.root)+1
            heappush(new, (d, e))
    h[index] = new


def add_neighbors_heap(n, h, nodes):
    """
    add neighbors to heap
    """
    n.build_arbs()
    for index in range(n.K):
        add_neighbors_heap_index(n, h, index, nodes)


def add_neighbors_heap_index(n, h, index, nodes):
    """
    add neighbors to heap for a given index and nodes
    """
    ni = n.nodes_index(index)
    dist = nx.shortest_path_length(n.g, target=n.root)
    for v in nodes:
        if v not in ni:
            continue
        preds = sorted(n.g.predecessors(v), key=lambda k: random.random())
        d = n.shortest_path_length(index, v, n.root)+1
        stretch = d*1.0/dist[v]
        stretch = d
        for x in preds:
            if x not in ni and n.g[x][v]['attr'] == -1:
                heappush(h[index], (stretch, (x, v)))


def round_robin(g: nx.DiGraph, cut: bool = False, swap: bool = False, reset: bool = True) -> int:
    """
    Basic round robin implementation of constructing arborescences
    @param g - Graph
    @param cut - Check edge connectivity on unsused graph after every arb construction
    @param swap - Swap edges already used in arbs to possibly fix deadlocks
    @param reset - reset attrs
    @return Number of trees constructed
    """
    global swappy
    if reset:
        reset_arb_attribute(g)
    g.graph['k'] = edmond(g)
    n = Network(g, g.graph['k'], g.graph['root'])
    K = n.K
    h = []
    dist = []
    prepareDS(n, h, dist, reset)
    index = 0
    swaps = 0
    count = 0
    num = len(g.nodes())
    count = 0
    while n.num_complete_nodes() < num and count < K*num*num:
        count += 1
        if len(h[index]) == 0:
            if swap and trySwap(n, h, index):
                index = (index + 1) % K
                swaps += 1
                continue
            else:
                if swap:
                    print("1 couldn't swap for index ", index)
                return 0
        (d, e) = heappop(h[index])
        while e != None and n.g[e[0]][e[1]]['attr'] > -1:  # in used_edges:
            if len(h[index]) == 0:
                if swap and trySwap(n, h, index):
                    index = (index + 1) % K
                    swaps += 1
                    e = None
                    continue
                else:
                    if swap:
                        print("2 couldn't swap for index ", index)
                    g = n.g
                    return 0
            else:
                (d, e) = heappop(h[index])
        ni = n.nodes_index(index)
        condition = (e != None and e[0] not in ni and e[1] in ni)
        if cut:
            condition = condition and (
                K - index == 1 or TestCut(n.rest_graph(index), e[0], n.root) >= K-index-1)
        if condition:
            n.add_to_index(e[0], e[1], index)
            add_neighbors_heap_index(n, h, index, [e[0]])
            index = (index + 1) % K
    swappy.append(swaps)
    g = n.g
    l = get_arborescence_list(g)
    return len(l)


def RouteDetCirc(s: int, d: int, marked_g: nx.DiGraph) -> tuple[bool, int, int, list[tuple[int, int]]]:
    """
    Try to route a packet from s to d on graph marked_g
    @param s - source node
    @param d - destination node
    @param marked_g - graph. Arbs marked by setting "attr" of according edges
    @return failed, hops, switches, route
    """
    T: list[nx.DiGraph] = []
    fails = []
    for edge in marked_g.edges:
        data = marked_g.get_edge_data(*edge)
        attr = data['attr']
        if data['failed']:
            fails.append((edge[0], edge[1]))
        while attr > (len(T) - 1):
            T.append(nx.DiGraph())
            T[-1].add_nodes_from(marked_g.nodes)
        T[attr].add_edge(edge[0], edge[1])
    T = T[1:]
    if len(T) == 0:
        return True, 0, 0, []
    curT = 0
    detour_edges = []
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    k = len(T)
    while (s != d):
        while (s not in T[curT].nodes()) and switches < k*n:
            curT = (curT+1) % k
            switches += 1
        if switches >= k*n:
            return True, hops, switches, detour_edges
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            return True, hops, switches, detour_edges
        nxt = nxt[0]
        if (s, nxt) in fails:
            curT = (curT+1) % k
            switches += 1
        else:
            detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > n or switches > k*n:
            return True, hops, switches, detour_edges
    return False, hops, switches, detour_edges
