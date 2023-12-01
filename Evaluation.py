import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from RoutingAlgorithms import RoutingAlgo
from GraphGenerators import GraphGenerator
from FailureGenerators import FailureGenerator
import DrawGraph
from Info import *
import copy


class Aggregator:
    """
    Aggregator that aggregates a list of RouteBuild- and RoutingInfos to a single value for use in plotting.
    """

    def __init__(self, extract_method, name: str, ignore_failed=False, ignore_impossible=False, aggregate_func="average"):
        """
        Constructor.
        @param extract_method - the method used to extract data
        @param name - the name of the aggregation, shown in the plot
        @param ignore_failed - whether to ignore failed routing attempts
        @param ignore_impossible - whether to ignore routing attempts where the destination was unreachable
        @param aggregate_func - the function used to aggregate the extracted data. "average" and "ratio" are supported
        """

        self.name = name
        self.extract_method = extract_method
        self.ignore_failed = ignore_failed
        self.ignore_impossible = ignore_impossible
        self.aggregate_func = aggregate_func

    def get_y(self, infos: list[tuple[RouteBuildInfo, RoutingInfo]]) -> float:
        """
        Aggregates a list of RouteBuildInfo and RoutingInfo tuples to a single float value for plotting.
        @param infos - a list of tuples containing RouteBuildInfo and RoutingInfo objects
        @return - single float value
        """

        infos = [info for info in infos if (info[1].found or not self.ignore_failed) and (
            info[1].possible or not self.ignore_impossible)]
        if len(infos) == 0:
            return 0
        values = [self.extract_method(*info) for info in infos]
        if self.aggregate_func == 'average':
            return sum(values) / len(values)
        elif self.aggregate_func == 'ratio':
            return np.count_nonzero(values) / len(values) * 100

    def get_name(self) -> str:
        """
        @return Name of the aggregator to be shown in plots.
        """
        return self.name


Hops = Aggregator(lambda build, route: route.hops,
                  "Avg. hops", aggregate_func="average")
Stretch = Aggregator(lambda build, route: route.stretch,
                     "Avg. stretch", True, aggregate_func="average")
Runtime = Aggregator(lambda build, route: build.runtime +
                     route.runtime, "Avg. runtime [s]", aggregate_func="average")
Packetloss = Aggregator(lambda build, route: not route.found,
                        "% of packets lost", False, True, aggregate_func="ratio")
FailedEdges = Aggregator(lambda build, route: len([1 for (u, v, d) in build.graph.edges(
    data=True) if d["failed"]]), "failed edges", aggregate_func="average")
EdgesUsable = Aggregator(lambda build, route: get_used_edges(
    build), "% of edges usable in routing", aggregate_func="average")
Loops = Aggregator(lambda build, route: route.loop,
                   "% of infinite loops", aggregate_func="ratio")
NumberOfArbs = Aggregator(lambda build, route: max([build.graph.get_edge_data(*edge)['attr'] for edge in build.graph.edges]), "number of arbs", aggregate_func="average")
Possible = Aggregator(lambda build, route: route.possible, "%of cases where dest. is reachable", aggregate_func="ratio")

UsedReverseRatio = Aggregator(
    lambda build, route: route.additional['ReverseEdgesUsed'] > 0,  "% of routes using reverse edges", True, True, aggregate_func="ratio")
UsedReverseAvg = Aggregator(
    lambda build, route: route.additional['ReverseEdgesUsed'],  "reverse edges used", True, True, aggregate_func="average")

def usable_a_links(build: RouteBuildInfo, route) -> float:
    num_alinks = len([e for e in build.graph.edges(data=True) if e[2]['link_type'] == "a-link"])
    if num_alinks != 0:
        return len([e for e in build.graph.edges(data=True) if e[2]['link_type'] == "a-link" and e[2]["normal_a_link"]])/num_alinks*100
    else:
        return 0
    
def a_links(build: RouteBuildInfo, route) -> float:
    num_edges = len(build.graph.edges)
    if num_edges != 0:
        return len([e for e in build.graph.edges(data=True) if e[2]['link_type'] == "a-link"])/num_edges*100
    else:
        return 0

def down_links(build: RouteBuildInfo, route) -> float:
    label_size, label, node_weight, down_links, A_links, up_links,_,_ = build.graph.graph["precomp"]
    num_edges = len(build.graph.edges)
    if num_edges != 0:
        return sum([len(s) for s in down_links.values()])/num_edges*100
    else:
        return 0

def up_links(build: RouteBuildInfo, route) -> float:
    label_size, label, node_weight, down_links, A_links, up_links,_,_ = build.graph.graph["precomp"]
    num_edges = len(build.graph.edges)
    if num_edges != 0:
        return sum([len(s) for s in up_links.values()])/num_edges*100
    else:
        return 0

def routed_a_links(build: RouteBuildInfo, route: RoutingInfo) -> float:
    path = route.route_taken
    path_len = len(path)
    if path_len != 0:
        return len([e for e in path if build.graph[e[0]][e[1]]["link_type"] == "a-link"])/path_len*100
    else:
        return 0
    
def routed_up_links(build: RouteBuildInfo, route: RoutingInfo) -> float:
    path = route.route_taken
    path_len = len(path)
    if path_len != 0:
        return len([e for e in path if build.graph[e[0]][e[1]]["link_type"] == "up-link"])/path_len*100
    else:
        return 0

def routed_down_links(build: RouteBuildInfo, route: RoutingInfo) -> float:
    path = route.route_taken
    path_len = len(path)
    if path_len != 0:
        return len([e for e in path if build.graph[e[0]][e[1]]["link_type"] == "down-link"])/path_len*100
    else:
        return 0

UsableALinks = Aggregator(usable_a_links, "% of a-links usable", False, False, aggregate_func="average")
ALinks = Aggregator(a_links, "a-link %", False, False, aggregate_func="average")
DownLinks = Aggregator(down_links, "downlink %", False, False, aggregate_func="average")
UpLinks = Aggregator(up_links, "uplink %", False, False, aggregate_func="average")
ALinksRouted = Aggregator(routed_a_links, "% of routed edges being a-links", False, False, aggregate_func="average")
DownLinksRouted = Aggregator(routed_down_links, "% of routed edges being down-links", False, False, aggregate_func="average")
UpLinksRouted = Aggregator(routed_up_links, "% of routed edges being up-links", False, False, aggregate_func="average")

def get_used_edges(build: RouteBuildInfo) -> float:
    """
    % of edges actually usable in routing. Usable edges are edges where attr is != 0
    """
    try:
        return len([e for e in build.graph.edges(data=True) if e[2]['attr'] != 0])/len(build.graph.edges)*100
    except:
        return 0


def eval(iterations: int, x_attr: str, x_values: list[float], y_attrs: list[Aggregator], graph_gen: GraphGenerator, fail_gen: FailureGenerator, routing_algos: list[RoutingAlgo], draw=False, draw_route=False, filename=None):
    """
    Evaluate the performance of routing algorithms on a generated graph with failures.
    @param iterations - How many iterations to simulate per x val
    @param x_attr - Attribute to change to the different x vals. Insert as kwarg into graph_gen, fail_gen and the routing_algos
    @param x_values - Values to set x to
    @param y_attrs - Aggregators to transform routing results into y-values
    @param graph_gen - Graph generator
    @param fail_gen - Failure generator
    @param routing_algo - List of routing algos to compare
    @param draw - Draw and display the result of build_route, showing the routes constructed (default False)
    @param draw_route -  Draw and display the result of route, showing the route taken (default False)
    @param filename - File to save the plots to (default None -> unsaved)
    """
    results = []  # [algo][x][iteration]
    results_aggregated = []  # [agg][algo][x] -> float
    for _ in routing_algos:
        results.append([])
        for _ in range(len(x_values)):
            results[-1].append([])
    for _ in y_attrs:
        results_aggregated.append([])
        for _ in routing_algos:
            results_aggregated[-1].append([])
            for _ in x_values:
                results_aggregated[-1][-1].append(.0)
    for x_ind, x in enumerate(x_values):
        print(f"x: {x}")
        for i in range(iterations):
            print(f"Iteration {i}")
            graph_gen.set_kwarg(**{x_attr: x})
            s, d, g = graph_gen.generate()
            fail_gen.set_kwarg(**{x_attr: x})
            fail_gen.generate(g)
            g_copy = copy.deepcopy(g)
            fails = []
            for edge in g_copy.edges:
                data = g_copy.get_edge_data(*edge)
                if data['failed']:
                    fails.append((edge[0], edge[1]))
            g_copy.remove_edges_from(fails)
            possible = nx.has_path(g_copy, s, d)
            try:
                shortest = nx.shortest_path_length(g, s, d)
            except:
                shortest = 0
            for algo_idx, algo in enumerate(routing_algos):
                # Shortcut - use only if impossible attempts are ignored in used Aggregators
                # if not possible:
                # ri = RoutingInfo(False, -1, [])
                # ri.possible = False
                # results[algo_idx][x_ind].append(ri)
                # continue
                algo.set_kwarg(**{x_attr: x})
                g_copy = copy.deepcopy(g)
                start = time.time()
                build_info = algo.build_routes(g_copy, s, d)
                build_info.runtime = time.time() - start
                marked_g = build_info.graph
                if draw:
                    DrawGraph.draw(marked_g)
                marked_g_copy = copy.deepcopy(marked_g)
                start = time.time()
                # try:   # Output graphs crashing the routing
                routing_info = algo.route(marked_g_copy, s, d)
                # except:
                # print(s, d, list(g.edges))
                # exit()
                routing_info.runtime = time.time() - start
                routing_info.possible = possible
                routing_info.stretch = routing_info.hops - shortest
                if draw_route:
                    DrawGraph.draw(marked_g, hops=routing_info.hops,
                                   routed_paths=routing_info.route_taken)
                # check_routing(build_info, routing_info)
                results[algo_idx][x_ind].append((build_info, routing_info))
        for algo_idx, algo in enumerate(routing_algos):
            for y_idx, y_attr in enumerate(y_attrs):
                results_aggregated[y_idx][algo_idx][x_ind] = y_attr.get_y(
                    results[algo_idx][x_ind])
            results[algo_idx][x_ind] = None  # Delete object
    # Plot results
    for y_idx, y_attr in enumerate(y_attrs):
        y_values = results_aggregated[y_idx]
        if filename != None:
            with open(filename + f"_{y_idx}.txt", "w") as f:
                f.write(str(y_values))
        else:
            print(y_values)
        name = y_attr.get_name()
        patch_labels = [algo.get_name() for algo in routing_algos]
        plot_eval(x_values, y_values, x_attr, name, patch_labels,
                  filename=filename + f"_{y_idx}" if filename != None else filename)


def eval_fr(iterations: int, x_attr: str, x_values: list[float], y_attrs: list[Aggregator], graph_gen: GraphGenerator, fail_gen: FailureGenerator, routing_algos: list[RoutingAlgo], draw=False, draw_route=False, filename=None):
    """
    Evaluate the performance of routing algorithms on a generated graph with failures. 
    !!! This version is more performant than eval(), because graphs aren't regenerated for every x_val. Thus, x_attrs affecting the routing algos or graph generation are not supported. Also aggregators using RouteBuildInfos may give wrong results.
    !!! Undirected graphs can give wrong results too, if they are transformed to directed graphs for the BuildInfo.
    @param iterations - How many iterations to simulate per x val
    @param x_attr - Attribute to change to the different x vals. Insert as kwarg into graph_gen, fail_gen and the routing_algos
    @param x_values - Values to set x to
    @param y_attrs - Aggregators to transform routing results into y-values
    @param graph_gen - Graph generator
    @param fail_gen - Failure generator
    @param routing_algo - List of routing algos to compare
    @param draw - Draw and display the result of build_route, showing the routes constructed (default False)
    @param draw_route -  Draw and display the result of route, showing the route taken (default False)
    @param filename - File to save the plots to (default None -> unsaved)
    """
    results = []  # [algo][x][iteration]
    for i in range(len(routing_algos)):
        results.append([])
        for _ in range(len(x_values)):
            results[i].append([])
    for i in range(iterations):
        marked_graphs = []
        print(f"{i+1}/{iterations}")
        s, d, g = graph_gen.generate()
        for algo_idx, algo in enumerate(routing_algos):
            marked_graphs.append(algo.build_routes(copy.deepcopy(g), s, d))
        for x_ind, x in enumerate(x_values):
            marked_graphs_xind = []
            for algo_idx, algo in enumerate(routing_algos):
                marked_graphs_xind.append(
                    copy.deepcopy(marked_graphs[algo_idx]))
            fail_gen.set_kwarg(**{x_attr: x})
            g_copy = copy.deepcopy(g)
            fail_gen.generate(g_copy)
            fails = []
            for edge in g_copy.edges:
                data = g_copy.get_edge_data(*edge)
                if data['failed']:
                    fails.append((edge[0], edge[1]))
                    for algo_idx, _ in enumerate(routing_algos):
                        marked_graphs_xind[algo_idx].graph[edge[0]
                                                           ][edge[1]]["failed"] = True
            g_copy.remove_edges_from(fails)
            possible = nx.has_path(g_copy, s, d)
            try:
                shortest = nx.shortest_path_length(g_copy, s, d)
            except:
                shortest = 0
            for algo_idx, algo in enumerate(routing_algos):
                build_info = marked_graphs_xind[algo_idx]
                #try:
                routing_info = algo.route(build_info.graph, s, d)
                '''except:
                    fails = []
                    for edge in build_info.graph.edges:
                        data = build_info.graph.get_edge_data(*edge)
                        if data['failed']:
                            fails.append((edge[0], edge[1]))
                    print(build_info.graph.edges())
                    print(s)
                    print(d)
                    exit(fails)'''
                routing_info.possible = possible
                routing_info.stretch = routing_info.hops - shortest
                results[algo_idx][x_ind].append((build_info, routing_info))
    # Plot results
    for i, y_attr in enumerate(y_attrs):
        y_values = [[y_attr.get_y(r) for r in algo] for algo in results]
        if filename != None:
            with open(filename + f"_{i}.txt", "w") as f:
                f.write(str(y_values))
        else:
            print(y_values)
        name = y_attr.get_name()
        patch_labels = [algo.get_name() for algo in routing_algos]
        plot_eval(x_values, y_values, x_attr, name, patch_labels,
                  filename=filename + f"_{i}" if filename != None else filename)


def plot_eval(x_values: list[float], y_values: list[list[float]], x_label: str, y_label: str, patch_labels: list[str], filename: str = None):
    """
    Displays the result plot.
    @param x_values: A list of floats representing the x-axis values. Values represent the changed parameter.
    @param y_values: A list of lists of floats representing the y-axis values. Each list represents one routing algorithm.
    @param x_label: x-axis label (changed param)
    @param y_label: y-axis label (aggregated measure)
    @param patch_labels: names of the routing algos measured
    @param filename - File to save the plot to (default None -> unsaved)
    """
    colors = [("blue", "skyblue"), ("red", "darkred"), ("green",
                                                        "lightgreen"), ("magenta", "magenta"), ("yellow", "lightyellow")]
    patches = []
    plt.cla()
    for i, y_vals in enumerate(y_values):
        plt.plot(x_values, y_vals, marker='o',
                 markerfacecolor=colors[i][0], markersize=6, color=colors[i][1], linewidth=4)
        patches.append(mpatches.Patch(
            color=colors[i][1], label=patch_labels[i]))
    plt.title(f"{x_label} - {y_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=patches)
    # plt.yscale("log")
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename + ".png")


def check_routing(rbi: RouteBuildInfo, ri: RoutingInfo):
    """
    Check if route described in rbi and ri is valid (e.g. uses coherent, non-failed edges on the graph)
    @param rbi: RouteBuildInfo
    @param ri: RoutingInfo
    @return True if valid
    """
    last_v = ri.route_taken[0][0] if len(ri.route_taken) != 0 else 0
    for u, v in ri.route_taken:
        # No jumping
        if u != last_v:
            print("JUMPED")
            return False
        last_v = v
        # No non-existent edges
        if not rbi.graph.has_edge(u, v):
            print("USED NON-EXISTENT")
            return False
        # No failed edges
        if rbi.graph[u][v]["failed"]:
            print("USED FAILED")
            return False
    return True
