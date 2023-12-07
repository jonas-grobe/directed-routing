Source code for experiments in our paper "Local Fast Failover Routing on Directed Networks"

Install: `pip install -r requirements.txt`
Run: `py Experiments.py`

Code partially based on:
https://github.com/oliver306/TREE (Oliver Schweiger, Klaus-Tycho Foerster, Stefan Schmid)  
https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/tree/master (Klaus-Tycho Foerster, Andrzej Kamisinski, Yvonne-Anne Pignolet, Stefan Schmid, Gilles Tredan)

# Graph Generation
`**kwargs -> tuple[int, int, nx.Graph] (start node, target node, graph)`
```python
@graph_gen
@require_args(["n", "p", "directed"])
def erdos_renyi(**kwargs) -> tuple[int, int, nx.DiGraph]:
    n = kwargs.get("n")
    p = kwargs.get("p")
    directed = kwargs.get("directed")
    g = nx.erdos_renyi_graph(n, p, directed=directed)
    nodes = list(g.nodes)
    s, d = random.sample(nodes, 2)
    return s, d, g
```

# Failure Generation
`**kwargs -> list[tuple[int, int]] (list of edges)`
```python
@fail_gen
@require_args(["n_failed_edges"])
def n_random_failure(g: nx.Graph, **kwargs):
    n_failed = kwargs.get("n_failed_edges")
    failed_edges = random.sample(list(g.edges()), n_failed)
    return failed_edges
```

# Routing Algorithms
Implemented as class inheriting RoutingAlgo:
```python
build_routes(self, g: nx.Graph, s: int, d: int) -> RouteBuildInfo # Routing precomputation, mark routes in the graph
route(self, marked_g: nx.Graph, s: int, d: int) -> RoutingInfo # Simulate routing and return results
get_name(self) -> str # The name of the algorithm (used in plots)
```
`RouteBuildInfo` and `RoutingInfo` can be found in Info.py

# Aggregators
Aggregators transform routing results into the final plot y-axis values.
```python
Hops = Aggregator(lambda build, route: route.hops, "Avg. hops", aggregate_func="average")
Packetloss = Aggregator(lambda build, route: not route.found, "% of packets lost", False, True, aggregate_func="ratio")
```
First optional bool arg: ignore failed routing attempts  
Second optional bool arg: ignore routing attempts where destination was unreachable

# Evaluation
```python
g_gen = erdos_renyi(n=20, p=0.8, directed=True)
f_gen = random_failure()
algos = [EDP(), Bonsai("random")]
Evaluation.eval(100, "failure_rate", [.1, .3, .5, .7], [Hops, Packetloss], g_gen, f_gen, algos)
```
This would output two plots (Hops plot and packetloss plot). If you are working with directed graphs and change no parameters regarding graph generation (which is the case in the given example), you can use eval_fr with the same parameters for faster simulation. It runs the graph generation and routing precomputation only once and only regenerates the failures for every x-value.
