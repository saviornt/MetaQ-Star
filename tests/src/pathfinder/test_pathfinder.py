# ./src/test_astar_pathfinder.py

import asyncio
import random
import logging
import cProfile
import pstats
import io
import os
import json

from pathfinder import AStarPathfinder, PathfinderConfig, PathfinderMetrics, log_metrics
from cache_management import CacheManager, CacheConfig
from resources import ResourceConfig, ResourceManager

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
)

def generate_q_values(size: int, num_entries: int):
    """Generate random Q-values for testing."""
    q_values = {}
    for _ in range(num_entries):
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        q_values[(x, y)] = {
            "action_a": random.uniform(0, 10),
            "action_b": random.uniform(0, 10),
            "action_c": random.uniform(0, 10)
        }
    return q_values

async def simulate_q_learning_updates(q_values, size, num_updates=1000):
    """Simulate Q-learning updates by randomly modifying Q-values."""
    for _ in range(num_updates):
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        if (x,y) not in q_values:
            q_values[(x,y)] = {}
        q_values[(x, y)]["action_a"] = random.uniform(0, 10)

async def run_single_test(pathfinder: AStarPathfinder, start, goal):
    """Run a single pathfinding test and log the result."""
    try:
        path, metrics = await pathfinder.bidirectional_a_star(start, goal)
        if path:
            logging.info(f"Path found from {start} to {goal}, length={len(path)}")
        else:
            logging.info(f"No path found or pruned from {start} to {goal}")
        return metrics
    except Exception as e:
        logging.exception(f"Error during pathfinding from {start} to {goal}")
        return None

async def main(complexity: int):
    """Main test function with profiling."""
    size = 100 * complexity
    num_q_entries = 2000 * complexity
    q_update_count = 200 * complexity

    logging.info(f"Running test with complexity={complexity}, size={size}, num_q_entries={num_q_entries}, q_update_count={q_update_count}")

    q_values = generate_q_values(size, num_q_entries)

    # Configuration setup
    pathfinder_config = PathfinderConfig(
        w1=1.0,
        w2=1.0,
        allow_diagonal=True,
        concurrent_expansions=20,
        initial_g_cost=0.0,
        initial_f_cost=0.0,
        grid_size=(size, size)  # Added grid_size
    )
    metrics = PathfinderMetrics()

    # Initialize CacheConfig
    cache_config = CacheConfig(
        use_redis=False,  # Set to True if you want to use Redis
        redis_url="redis://localhost",  # Specify if using Redis
        cache_maxsize=5000,
        cache_ttl=300.0
    )
    cache_manager = CacheManager(config=cache_config)

    resource_config = ResourceConfig(
        observation_period=10.0,
        check_interval=1.0,
        rolling_window_size=20,
        target_utilization=0.9,
        enabled=False
    )
    resource_manager = ResourceManager(config=resource_config)

    # Initialize the pathfinder
    pathfinder = AStarPathfinder(
        q_values=q_values,
        config=pathfinder_config,
        metrics=metrics,
        cache_config=cache_config,
        cache_manager=cache_manager,
        resource_manager=resource_manager
    )

    # Define pathfinding scenarios
    scenarios = [
        ((0,0), (size-1, size-1)),
        ((10,10), (size//2, size//2)),
        ((size//4, size//4), (size-10, size-10))
    ]

    # Profile the pathfinding runs
    profiler = cProfile.Profile()
    profiler.enable()

    # Run initial pathfinding tests
    tasks = [run_single_test(pathfinder, start, goal) for start, goal in scenarios]
    initial_metrics = await asyncio.gather(*tasks)

    # Simulate Q-learning updates
    logging.info("Simulating Q-value updates...")
    await simulate_q_learning_updates(q_values, size, num_updates=q_update_count)

    # Reprocess Q-values after updates
    pathfinder.q_array = pathfinder.preprocess_q_values(q_values, pathfinder.config.grid_size)

    # Run pathfinding tests again after Q-learning updates
    tasks = [run_single_test(pathfinder, start, goal) for start, goal in scenarios]
    updated_metrics = await asyncio.gather(*tasks)

    # Disable the profiler and save profiling results
    profiler.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(50)  # Adjust the number to print more or fewer functions

    # Save profiling results to a file
    profiling_dir = "profiling_results"
    os.makedirs(profiling_dir, exist_ok=True)
    profiling_file = os.path.join(profiling_dir, f"profile_complexity_{complexity}.txt")
    with open(profiling_file, "w") as f:
        f.write(s.getvalue())

    logging.info(f"Profiling complete. Results saved to {profiling_file}")

    # Save metrics to a separate log file
    metrics_dir = "metrics_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"metrics_complexity_{complexity}.json")

    # Aggregate metrics
    aggregated_metrics = {
        "initial_run": [],
        "updated_run": []
    }

    # Collect initial run metrics
    for metric in initial_metrics:
        if metric:
            aggregated_metrics["initial_run"].append(metric.to_dict() if hasattr(metric, 'to_dict') else metric.__dict__)

    # Collect updated run metrics
    for metric in updated_metrics:
        if metric:
            aggregated_metrics["updated_run"].append(metric.to_dict() if hasattr(metric, 'to_dict') else metric.__dict__)

    # Write metrics to JSON file
    with open(metrics_file, "w") as f:
        json.dump(aggregated_metrics, f, indent=4)

    logging.info(f"Metrics logged to {metrics_file}")

    # Clean up resource manager
    resource_manager.close()

    logging.info("Final real-world test complete.")

if __name__ == "__main__":
    # Adjust the complexity level as desired (e.g., 1 for small, 2 for medium, etc.)
    complexity_level = 2
    for i in range(1, complexity_level + 1):
        asyncio.run(main(i))
