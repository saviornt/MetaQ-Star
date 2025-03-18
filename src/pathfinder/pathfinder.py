# ./src/pathfinder/pathfinder.py

"""
This module implements the A* pathfinding algorithm, specifically a bidirectional variant, 
designed for grid-based environments.  It leverages asynchronous operations and caching 
to optimize performance.

The core component is the `AStarPathfinder` class, which manages the search process, 
utilizing a preprocessed Q-value representation for heuristic calculations.  It supports 
concurrent node expansions and incorporates resource management to handle potential 
constraints.  The algorithm employs a bidirectional search strategy, expanding from both 
the start and goal nodes simultaneously, to efficiently find a path.  Caching is used to 
store and retrieve previously calculated G-values, further speeding up the search. 

The module also provides functionality for recording pathfinding metrics, such as 
execution time, number of node expansions, nodes visited, and path length.  It supports 
both in-memory and Redis based caching.
"""

import asyncio
import heapq
import time
from typing import Dict, Tuple, Any, List, Optional
import logging
from numba import njit
import numpy as np

from pydantic.dataclasses import dataclass
from pydantic import Field

from cache_manager.cache_manager import CacheManager, CacheConfig
from src.utils.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

# Pre-defined direction vectors
DIRECTIONS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRECTIONS_8 = DIRECTIONS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

@dataclass
class PathfinderConfig:
    w1: float = Field(
        default=1.0,
        description="Weight for Q-value heuristic component"
    )
    w2: float = Field(
        default=1.0,
        description="Weight for distance-based heuristic component"
    )
    grid_size: Tuple[int, int] = Field(
        default=(100, 100),
        description="Grid size as (width, height)"
    )
    allow_diagonal: bool = Field(
        default=True,
        description="Allow diagonal movement"
    )
    concurrent_expansions: int = Field(
        default=50,
        description="Number of expansions to run concurrently"
    )
    initial_g_cost: float = Field(
        default=0.0,
        description="Initial g-cost for start/goal nodes in the search"
    )
    initial_f_cost: float = Field(
        default=0.0,
        description="Initial f-cost when pushing start/goal onto priority queues"
    )

@dataclass
class PathfinderMetrics:
    pathfinding_times: List[float] = Field(
        default_factory=list,
        description="List of pathfinding times in seconds"
    )
    expansions_counts: List[int] = Field(
        default_factory=list,
        description="List of expansion counts"
    )
    nodes_visited_counts: List[int] = Field(
        default_factory=list,
        description="List of nodes visited counts"
    )
    path_lengths: List[int] = Field(
        default_factory=list,
        description="List of path lengths"
    )

@njit
def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (1.4142135623730951 - 1) * min(dx, dy)

@njit
def move_cost(current: Tuple[int,int], neighbor: Tuple[int,int]) -> float:
    # Cost 1 for straight moves, sqrt(2) for diagonals
    return 1.0 if (current[0] == neighbor[0] or current[1] == neighbor[1]) else 1.4142135623730951

@njit
def heuristic(state_x: int, state_y: int, goal_x: int, goal_y: int, q_array: np.ndarray, w1: float, w2: float) -> float:
    q_val = q_array[state_x, state_y]
    dx = abs(state_x - goal_x)
    dy = abs(state_y - goal_y)
    dist = max(dx, dy) + (1.4142135623730951 - 1) * min(dx, dy)
    return w1 * q_val + w2 * dist

def get_neighbors(node: Tuple[int,int], allow_diagonal: bool = True, grid_size: Tuple[int,int] = (100, 100)) -> List[Tuple[int,int]]:
    x, y = node
    max_x, max_y = grid_size
    directions = DIRECTIONS_8 if allow_diagonal else DIRECTIONS_4
    neighbors = [(x + dx, y + dy) for dx, dy in directions]
    # Filter out-of-bounds neighbors
    return [n for n in neighbors if 0 <= n[0] < max_x and 0 <= n[1] < max_y]

def reconstruct_path(
    came_from_forward: Dict[Tuple[int,int], Tuple[int,int]],
    came_from_backward: Dict[Tuple[int,int], Tuple[int,int]],
    meeting_node: Tuple[int,int]
) -> List[Tuple[int,int]]:
    path = []
    current = meeting_node
    while current in came_from_forward:
        path.append(current)
        current = came_from_forward[current]
    path.reverse()

    current = meeting_node
    while current in came_from_backward:
        current = came_from_backward[current]
        path.append(current)
    return path

async def log_metrics(metrics: PathfinderMetrics):
    """Log aggregated metrics information."""
    if metrics.pathfinding_times:
        avg_time = sum(metrics.pathfinding_times) / len(metrics.pathfinding_times)
    else:
        avg_time = 0.0

    if metrics.expansions_counts:
        avg_expansions = sum(metrics.expansions_counts) / len(metrics.expansions_counts)
    else:
        avg_expansions = 0.0

    if metrics.nodes_visited_counts:
        avg_nodes = sum(metrics.nodes_visited_counts) / len(metrics.nodes_visited_counts)
    else:
        avg_nodes = 0.0

    if metrics.path_lengths:
        avg_path_length = sum(metrics.path_lengths) / len(metrics.path_lengths)
    else:
        avg_path_length = 0.0

    logger.info("=== Aggregated Pathfinder Metrics ===")
    logger.info(f"Average Pathfinding Time: {avg_time:.6f} seconds")
    logger.info(f"Average Expansions: {avg_expansions:.2f}")
    logger.info(f"Average Nodes Visited: {avg_nodes:.2f}")
    logger.info(f"Average Path Length: {avg_path_length:.2f}")

class AStarPathfinder:
    """
    Implements a bidirectional A* search algorithm for finding optimal paths in a grid-based environment.

    This class preprocesses Q-values into a NumPy array for efficient heuristic evaluation and supports
    asynchronous node expansion. It leverages caching (with both synchronous and asynchronous access)
    and resource management to control concurrent expansions and prevent resource overuse during the search.

    Parameters:
        q_values (Dict[Tuple[int, int], Dict[Any, float]]):
            A mapping from grid coordinates (x, y) to dictionaries of action costs. Each inner dictionary
            associates an action (or state transition) with a float cost.
        config (Optional[PathfinderConfig]):
            Configuration settings for the pathfinding algorithm, such as grid size, cost weights,
            initial costs, and concurrency settings. Defaults to a new PathfinderConfig instance if not provided.
        metrics (Optional[PathfinderMetrics]):
            An object used to record performance metrics (e.g., pathfinding time, number of node expansions,
            nodes visited, and path lengths). Defaults to a new PathfinderMetrics instance if not provided.
        cache_config (Optional[CacheConfig]):
            Configuration parameters for initializing a CacheManager when one is not provided.
        cache_manager (Optional[CacheManager]):
            An instance responsible for caching intermediate results (e.g., tentative costs) to optimize
            the search process. If not provided, a new CacheManager is created using the cache_config.
        resource_manager (Optional[ResourceManager]):
            An instance that monitors and manages system resources, enabling the algorithm to prune expansions
            if resources become constrained. Defaults to a new ResourceManager if not provided.

    Attributes:
        q_values:
            The original mapping of grid coordinates to their corresponding action cost dictionaries.
        config:
            The configuration parameters governing the search behavior.
        metrics:
            The object that collects performance metrics during pathfinding.
        cache_manager:
            The manager that handles caching operations (both synchronous and asynchronous) to store and
            retrieve node expansion costs.
        resource_manager:
            The manager that determines whether resource constraints require pruning the search.
        concurrent_expansions (int):
            Maximum number of node expansions allowed concurrently, derived from the configuration.
        _expansion_semaphore (asyncio.Semaphore):
            A semaphore to control and limit concurrent asynchronous node expansions.
        q_array (np.ndarray):
            A preprocessed NumPy array representation of the Q-values, where each cell contains the maximum
            action cost for that grid coordinate.

    Methods:
        preprocess_q_values(q_values: Dict[Tuple[int, int], Dict[Any, float]], grid_size: Tuple[int, int]) -> np.ndarray:
            Converts the provided Q-values dictionary into a NumPy array of shape `grid_size`, where each cell
            holds the maximum cost from its corresponding action cost dictionary.
        
        bidirectional_a_star(start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], PathfinderMetrics]:
            Asynchronously performs a bidirectional A* search from the start to the goal coordinate.
            The method concurrently expands nodes from both directions until the searches meet or resource constraints
            force pruning. Returns a tuple with the found path (or None if no path is found) and the collected metrics.
        
        _expand_nodes(current: Tuple[int, int],
                      g_map: Dict[Tuple[int, int], float],
                      came_from: Dict[Tuple[int, int], Tuple[int, int]],
                      open_list: List[Tuple[float, Tuple[int, int]]],
                      goal: Tuple[int, int],
                      direction: str) -> Tuple[int, Set]:
            Asynchronously expands the neighboring nodes of the current node. This method calculates tentative
            costs, updates the path, and interacts with the cache manager (using synchronous or asynchronous methods
            based on configuration). It returns the number of expansions performed and a set of visited neighbor nodes.
        
        _record_metrics(start_time: float, expansions: int, nodes_visited: int, path_length: int):
            Records performance metrics including elapsed time, number of expansions, nodes visited, and the length
            of the path (if found). The metrics are stored in the `metrics` attribute.
        
        pathfinder_cache_invalidation():
            Invalidates the cache via the CacheManager by calling its clear method (asynchronously or synchronously
            based on its configuration). Logs the result of the cache invalidation process.
    
    Example:
        >>> pathfinder = AStarPathfinder(q_values, config, metrics, cache_config, cache_manager, resource_manager)
        >>> path, metrics = await pathfinder.bidirectional_a_star((0, 0), (5, 5))
    """
    def __init__(
        self, 
        q_values: Dict[Tuple[int,int], Dict[Any,float]],
        config: Optional[PathfinderConfig] = None,
        metrics: Optional[PathfinderMetrics] = None,
        cache_config: Optional[CacheConfig] = None,
        cache_manager: Optional[CacheManager] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        self.q_values = q_values
        self.config = config or PathfinderConfig()
        self.metrics = metrics or PathfinderMetrics()
        if cache_manager:
            self.cache_manager = cache_manager
        else:
            # Initialize CacheManager with provided cache_config
            self.cache_manager = CacheManager(config=cache_config)
        self.resource_manager = resource_manager or ResourceManager()

        self.concurrent_expansions = self.config.concurrent_expansions
        self._expansion_semaphore = asyncio.Semaphore(self.concurrent_expansions)

        # Preprocess Q-values into a NumPy array for faster access
        self.q_array = self.preprocess_q_values(self.q_values, self.config.grid_size)

    def preprocess_q_values(self, q_values: Dict[Tuple[int,int], Dict[Any,float]], grid_size: Tuple[int, int]) -> np.ndarray:
        q_array = np.zeros(grid_size, dtype=np.float32)
        for (x, y), actions in q_values.items():
            q_array[x, y] = max(actions.values())
        return q_array

    async def bidirectional_a_star(
        self,
        start: Tuple[int,int],
        goal: Tuple[int,int]
    ) -> Tuple[Optional[List[Tuple[int,int]]], PathfinderMetrics]:
        start_time = time.time()

        forward_open = []
        backward_open = []
        heapq.heappush(forward_open, (self.config.initial_f_cost, start))
        heapq.heappush(backward_open, (self.config.initial_f_cost, goal))

        g_forward = {start: self.config.initial_g_cost}
        g_backward = {goal: self.config.initial_g_cost}
        came_from_forward = {}
        came_from_backward = {}
        closed_forward = set()
        closed_backward = set()

        expansions = 0
        visited = set([start, goal])

        try:
            while forward_open and backward_open:
                if self.resource_manager.should_prune():
                    logger.info("Resources constrained, pruning expansions.")
                    break

                _, current_forward = heapq.heappop(forward_open)
                closed_forward.add(current_forward)

                _, current_backward = heapq.heappop(backward_open)
                closed_backward.add(current_backward)

                # Check if searches have met
                if current_forward in closed_backward:
                    path = reconstruct_path(came_from_forward, came_from_backward, current_forward)
                    self._record_metrics(start_time, expansions, len(visited), len(path))
                    return path, self.metrics
                if current_backward in closed_forward:
                    path = reconstruct_path(came_from_forward, came_from_backward, current_backward)
                    self._record_metrics(start_time, expansions, len(visited), len(path))
                    return path, self.metrics

                # Expand forward and backward concurrently
                expansions_fw, visited_fw = await self._expand_nodes(current_forward, g_forward, came_from_forward, forward_open, goal, direction='forward')
                expansions_bw, visited_bw = await self._expand_nodes(current_backward, g_backward, came_from_backward, backward_open, start, direction='backward')

                expansions += expansions_fw + expansions_bw
                visited.update(visited_fw)
                visited.update(visited_bw)

                await asyncio.sleep(0)

            # No path found or pruned
            self._record_metrics(start_time, expansions, len(visited), 0)
            return None, self.metrics
        finally:
            await self.cache_manager.clear()
            #pass

    async def _expand_nodes(
        self,
        current: Tuple[int,int],
        g_map: Dict[Tuple[int,int], float],
        came_from: Dict[Tuple[int,int], Tuple[int,int]],
        open_list: List[Tuple[float, Tuple[int,int]]],
        goal: Tuple[int,int],
        direction: str
    ):
        neighbors = get_neighbors(current, allow_diagonal=self.config.allow_diagonal, grid_size=self.config.grid_size)
        expansions_count = 0
        visited_nodes = set()
        async with self._expansion_semaphore:
            for neighbor in neighbors:
                # Synchronous cache access for in-memory cache
                if not self.cache_manager.use_redis:
                    tentative_g = g_map[current] + move_cost(current, neighbor)
                    cache_key = (neighbor, direction)
                    cached_cost = self.cache_manager.get_sync(cache_key)
                    if cached_cost is not None and cached_cost <= tentative_g:
                        continue

                    if neighbor not in g_map or tentative_g < g_map[neighbor]:
                        g_map[neighbor] = tentative_g
                        # Heuristic calculation
                        f_val = heuristic(
                            neighbor[0],
                            neighbor[1],
                            goal[0],
                            goal[1],
                            self.q_array,
                            self.config.w1,
                            self.config.w2
                        )
                        heapq.heappush(open_list, (f_val, neighbor))
                        came_from[neighbor] = current
                        self.cache_manager.set_sync(cache_key, tentative_g)
                        expansions_count += 1
                        visited_nodes.add(neighbor)
                else:
                    # Asynchronous cache access for Redis
                    tentative_g = g_map[current] + move_cost(current, neighbor)
                    cache_key = (neighbor, direction)
                    cached_cost = await self.cache_manager.async_get(cache_key)
                    if cached_cost is not None and cached_cost <= tentative_g:
                        continue

                    if neighbor not in g_map or tentative_g < g_map[neighbor]:
                        g_map[neighbor] = tentative_g
                        # Heuristic calculation
                        f_val = heuristic(
                            neighbor[0],
                            neighbor[1],
                            goal[0],
                            goal[1],
                            self.q_array,
                            self.config.w1,
                            self.config.w2
                        )
                        heapq.heappush(open_list, (f_val, neighbor))
                        came_from[neighbor] = current
                        await self.cache_manager.async_set(cache_key, tentative_g)
                        expansions_count += 1
                        visited_nodes.add(neighbor)
        return expansions_count, visited_nodes

    def _record_metrics(self, start_time: float, expansions: int, nodes_visited: int, path_length: int):
        end_time = time.time()
        elapsed = end_time - start_time
        self.metrics.pathfinding_times.append(elapsed)
        self.metrics.expansions_counts.append(expansions)
        self.metrics.nodes_visited_counts.append(nodes_visited)
        if path_length > 0:
            self.metrics.path_lengths.append(path_length)
        logger.info(f"Pathfinding finished in {elapsed:.6f} seconds. Expansions={expansions}, NodesVisited={nodes_visited}, PathLength={path_length}")
    
    def pathfinder_cache_invalidation(self):
        """
        Invalidate the cache using the CacheManager.clear() method.
        """
        try:
            if self.cache_manager.use_redis:
                asyncio.run(self.cache_manager.clear())
            else:
                self.cache_manager.clear()
            logger.info("Cache invalidated successfully.")
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")