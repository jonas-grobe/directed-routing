import GraphGenerators as gg
import FailureGenerators as fg
import RoutingAlgorithms as ra
import Evaluation
from DrawGraph import draw

f_05 = [.0, .1, .2, .3, .4, .5]
f_08 = f_05 + [.6, .7, .8]
f_09 = f_08 + [.9]

algos = [ra.Bonsai("greedy"), ra.Bonsai("round-robin"),
         ra.Grafting(), ra.KeepForwarding()]
metrics = [Evaluation.Packetloss, Evaluation.Stretch, Evaluation.Possible]


# Change graph generation params here!
gen = gg.wireless(n=30, area=50, min_range=10, max_range=20)

# Generate # draw single graph
# draw(gen.generate()[2])

# Experiments
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gen, fg.random_failure(), algos, filename="results/wireless_30_50_10_20")

gen = gg.wireless(n=40, area=80, min_range=15, max_range=25)
Evaluation.eval_fr(500, "failure_rate", f_05, metrics, gen, fg.random_failure(), algos, filename="results/wireless_50_80_20_35")
