import GraphGenerators as gg
import FailureGenerators as fg
import RoutingAlgorithms as ra
import Evaluation

f_05 = [.0, .1, .2, .3, .4, .5]
f_08 = f_05 + [.6, .7, .8]
f_09 = f_08 + [.9]

algos = [ra.Bonsai("greedy"), ra.Bonsai("round-robin"),
         ra.Grafting(), ra.KeepForwarding()]
metrics = [Evaluation.Packetloss, Evaluation.Stretch]

# All simulations in the paper
# Add parameter filename, to write results to file
# Add parameters draw/draw_route = True to draw graph

# Fig. 5 + 6
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.random_directed(
    n=25, d=6, p=0.1), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.random_directed(
    n=25, d=6, p=0.3), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.random_directed(
    n=25, d=6, p=0.75), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.erdos_renyi(
    n=25, p=0.35, directed=True), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.erdos_renyi(
    n=50, p=0.35, directed=True), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.erdos_renyi(
    n=25, p=0.6, directed=True), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.wireless(
    n=25, area=50, min_range=10, max_range=20), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.wireless(
    n=50, area=80, min_range=15, max_range=25), fg.random_failure(), algos)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gg.wireless(
    n=25, area=50, min_range=5, max_range=25), fg.random_failure(), algos)

# Fig. 8
Evaluation.eval_fr(500, "failure_rate", f_05, metrics,
                   gg.heathland(), fg.random_failure(), algos)


## More experiments, not included in paper

# Runtime
Evaluation.eval(50, "n", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [
                Evaluation.Runtime], gg.erdos_renyi(n=5, p=0.35, directed=True), fg.no_failure(), algos)

# Keep Forwarding
Evaluation.eval(100, "p", f_09 + [1], [Evaluation.ALinks, Evaluation.UsableALinks],
                gg.erdos_renyi(n=50, p=0, directed=True), fg.no_failure(), [ra.KeepForwarding(version=1)])
Evaluation.eval(100, "p", f_09 + [1], [Evaluation.DownLinks, Evaluation.ALinks, Evaluation.UpLinks],
                gg.erdos_renyi(n=50, p=0, directed=True), fg.no_failure(), [ra.KeepForwarding()])
Evaluation.eval(100, "failure_rate", f_08, [Evaluation.DownLinksRouted, Evaluation.ALinksRouted, Evaluation.UpLinksRouted],
                gg.erdos_renyi(n=50, p=0.35, directed=True), fg.random_failure(), [ra.KeepForwarding()])
Evaluation.eval(500, "failure_rate", f_05, [Evaluation.Packetloss, Evaluation.Stretch], gg.erdos_renyi(
    n=50, p=0.35, directed=True), fg.random_failure(), [ra.KeepForwarding(), ra.KeepForwarding(version=1)])

# Bonsai + Grafting
Evaluation.eval(500, "p", [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], [Evaluation.EdgesUsable, Evaluation.NumberOfArbs], gg.erdos_renyi(
    n=20, p=0, directed=True), fg.random_failure(failure_rate=0.5), [ra.Bonsai("greedy"), ra.Bonsai("round-robin"), ra.Grafting()])

# Erdos-Renyi graph generation with different parameters
Evaluation.eval(500, "n", [10, 20, 30, 40, 50], [Evaluation.Packetloss, Evaluation.Stretch], gg.erdos_renyi(
    n=25, p=0.35, directed=True), fg.random_failure(failure_rate=0.5), algos)
Evaluation.eval(500, "p", [0.1, 0.3, 0.5, 0.7, 0.9], [Evaluation.Packetloss, Evaluation.Stretch], gg.erdos_renyi(
    n=25, p=0.35, directed=True), fg.random_failure(failure_rate=0.5), algos)

# Directed graph generation with different parameters
Evaluation.eval(500, "n", [10, 20, 30, 40, 50], [Evaluation.Packetloss, Evaluation.Stretch],
                gg.random_directed(n=50, d=6, p=0.5), fg.random_failure(failure_rate=0.5), algos)
Evaluation.eval(500, "d", [2, 4, 6, 8, 10, 12, 14, 16], [Evaluation.Packetloss, Evaluation.Stretch],
                gg.random_directed(n=50, d=6, p=0.5), fg.random_failure(failure_rate=0.5), algos)
