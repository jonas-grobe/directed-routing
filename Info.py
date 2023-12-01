from dataclasses import dataclass, field
from typing import Any, Optional
import networkx as nx


@dataclass(slots=True)
class RoutingInfo:
    """
    Dataclass that represents information about a single routing attempt. It has the following attributes:
    - found: a boolean indicating whether the route was found or not
    - hops: number of hops taken to reach the destination
    - route_taken: path the package travelled
    - possible: optional, boolean indicating whether reaching the destination from the source is possible (default is False)
    - stretch: optional, stretch of the route (default is 0)
    - loop: optional, boolean indicating whether the routing encountered a loop or not (default is False)
    - runtime: optional, time taken to compute the routing (default is 0.0)
    - additional: optional, dict to store more info.
    runtime, possible and stretch are set in Evaluation.eval()
    """
    found: bool
    hops: int
    route_taken: list[int]
    possible: Optional[bool] = False
    stretch: Optional[int] = 0
    loop: Optional[bool] = False
    runtime: Optional[float] = 0.0
    additional: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class RouteBuildInfo:
    """
    Dataclass that represents information about a route building process. It has three attributes:
    - success: a boolean indicating whether the route building process was successful or not.
    - graph: Route building methods are allowed to modify the graph (e.g. settings edges ["attr"]) to store routes to be used in routing
    - runtime: optional, time taken to build the route (default is 0.0)
    - additional: optional, dict to store more info
    runtime is set in Evaluation.eval()
    """
    success: bool
    graph: nx.Graph  # Different paths marked with attr
    runtime: Optional[float] = 0.0
    additional: Optional[dict[str, Any]] = field(default_factory=dict)
