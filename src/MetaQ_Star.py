"""
MetaQ-Star: Main Module

This is the main entry point for the MetaQ-Star package, a machine learning framework
that uses a combined approach of MAML with Q-Learning that utilizes A* Pathfinding 
for reinforcement learning, with configurable storage options and hyperparameter optimization.

The module seamlessly integrates:
- MAML (Model-Agnostic Meta-Learning) for quick adaptation to new tasks
- Q-Learning with A* Pathfinding for efficient navigation and learning
- Optuna for hyperparameter optimization
- Ray for distributed computing and scaling
"""

import os
import logging
import asyncio
import datetime
import copy
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field

# Import core components
from optimizer.optimizer import Optimizer
from optimizer.optimizer_config import OptimizerConfig
from meta_learning.maml import MAMLAgent, MAMLConfig
from re_learning.q_learning import (
    QLearningAgent, 
    QLearningParams,
    HierarchicalAgentSystemConfig,
    AgentConfig,
    HierarchicalMultiAgentEnvironment
)
from pathfinder import AStarPathfinder, PathfinderConfig, PathfinderMetrics

# Import database components
from databases.db_config import DBConfig
from databases.mongo import MongoDBConnector
from databases.redis import RedisConnector
from databases.sqlite import SQLiteConnector
from databases.system_memory import SystemMemoryConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StorageType(str, Enum):
    """Enum for storage type selection."""
    CLOUD = "cloud"  # MongoDB + Redis
    LOCAL = "local"  # SQLite + System Memory

class DatasetType(str, Enum):
    """Enum for dataset type selection."""
    GRID_WORLD = "grid_world"
    MAZE = "maze"
    CUSTOM = "custom"

class MetaQStarConfig(BaseModel):
    """
    Main configuration for the MetaQ-Star system.
    
    This configuration class centralizes all settings for the system, including
    storage options, environment parameters, and optimizer configurations.
    
    Attributes:
        dataset_type (DatasetType): Type of dataset/environment to use
        storage_type (StorageType): Type of storage backend to use (cloud or local)
        optimizer_config (OptimizerConfig): Configuration for the optimizer
        maml_config (MAMLConfig): Configuration for MAML
        q_learning_config (QLearningParams): Configuration for Q-Learning
        pathfinder_config (PathfinderConfig): Configuration for A* Pathfinding
        system_config (Optional[HierarchicalAgentSystemConfig]): Configuration for agent system
        db_config (DBConfig): Configuration for database connections
        enable_checkpointing (bool): Whether to save checkpoints during training
        checkpoint_interval (int): Interval for saving checkpoints (in training steps)
        checkpoint_path (str): Path to save checkpoints
        device (str): Device to use for computation (cuda, mps, or cpu)
        grid_obstacles_percentage: float = Field(
            default=0.2,
            description="Percentage of grid cells that are obstacles (for grid world)"
        )
        maze_complexity: float = Field(
            default=0.7,
            description="Complexity of the maze (for maze environment)"
        )
    """
    dataset_type: DatasetType = Field(
        default=DatasetType.GRID_WORLD,
        description="Type of dataset/environment to use"
    )
    storage_type: StorageType = Field(
        default=StorageType.LOCAL,
        description="Type of storage backend to use"
    )
    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Configuration for the optimizer"
    )
    maml_config: MAMLConfig = Field(
        default_factory=MAMLConfig,
        description="Configuration for MAML"
    )
    q_learning_config: QLearningParams = Field(
        default_factory=QLearningParams,
        description="Configuration for Q-Learning"
    )
    pathfinder_config: PathfinderConfig = Field(
        default_factory=PathfinderConfig,
        description="Configuration for A* Pathfinding"
    )
    system_config: Optional[HierarchicalAgentSystemConfig] = Field(
        default=None,
        description="Configuration for hierarchical agent system"
    )
    db_config: DBConfig = Field(
        default_factory=DBConfig,
        description="Configuration for database connections"
    )
    enable_checkpointing: bool = Field(
        default=True,
        description="Whether to save checkpoints during training"
    )
    checkpoint_interval: int = Field(
        default=100,
        description="Interval for saving checkpoints (in training steps)"
    )
    checkpoint_path: str = Field(
        default="./checkpoints",
        description="Path to save checkpoints"
    )
    device: str = Field(
        default="cuda",  # Default to CUDA if available (as per tech stack)
        description="Device to use for computation (cuda, mps, or cpu)"
    )
    grid_obstacles_percentage: float = Field(
        default=0.2,
        description="Percentage of grid cells that are obstacles (for grid world)"
    )
    maze_complexity: float = Field(
        default=0.7,
        description="Complexity of the maze (for maze environment)"
    )

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

class MetaQStar:
    """
    Main class for the MetaQ-Star system.
    
    This class integrates MAML with Q-Learning that utilizes A* Pathfinding,
    providing a meta-reinforcement learning approach that can quickly adapt
    to new navigation tasks.
    
    Core components:
    - Meta-learning (MAML) for quick adaptation to new tasks
    - Q-Learning for reinforcement learning
    - A* Pathfinding for efficient navigation
    - Optuna for hyperparameter optimization
    - Ray for distributed computing and scaling
    """
    
    def __init__(self, config: Optional[MetaQStarConfig] = None):
        """
        Initialize the MetaQ-Star system.
        
        Args:
            config (Optional[MetaQStarConfig]): Configuration for the system.
                If None, default configuration will be used.
        """
        self.config = config or MetaQStarConfig()
        
        # Create a default system config if not provided
        if not self.config.system_config:
            self.config.system_config = self._create_default_system_config()
        
        # Initialize storage based on config
        self.db_connectors = self._init_storage()
        
        # Initialize optimizer for hyperparameter optimization
        self.optimizer = Optimizer(self.config.optimizer_config)
        
        # Initialize components (will be created when needed)
        self.maml_agent = None
        self.pathfinder = None
        self.q_learning_agent = None
        self.custom_environment = None
        
        # Store registered environments and tasks
        self.environments = {}
        self.task_environments = {}
        
        # Ensure checkpoint directory exists
        if self.config.enable_checkpointing:
            os.makedirs(self.config.checkpoint_path, exist_ok=True)
    
    def _create_default_system_config(self) -> HierarchicalAgentSystemConfig:
        """Create a default system configuration."""
        # Create default agent config
        agent_config = AgentConfig(
            agent_id="agent1",
            q_learning_config=self.config.q_learning_config
        )
        
        # Create default system config
        system_config = HierarchicalAgentSystemConfig(
            agents_config=[agent_config],
            grid_size=self.config.q_learning_config.state_space,
            action_space=self.config.q_learning_config.action_space,
            pathfinder_config=self.config.pathfinder_config
        )
        
        return system_config
    
    def _init_storage(self) -> Dict[str, Any]:
        """
        Initialize storage backends based on configuration.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized database connectors.
        """
        connectors = {}
        
        if self.config.storage_type == StorageType.CLOUD:
            # Initialize cloud storage (MongoDB + Redis)
            connectors['document_db'] = MongoDBConnector(self.config.db_config.mongo)
            connectors['cache'] = RedisConnector(self.config.db_config.redis)
            logger.info("Initialized cloud storage (MongoDB + Redis)")
        else:
            # Initialize local storage (SQLite + System Memory)
            connectors['document_db'] = SQLiteConnector(self.config.db_config.sqlite)
            connectors['cache'] = SystemMemoryConnector()
            logger.info("Initialized local storage (SQLite + System Memory)")
        
        return connectors
    
    def _init_components(self) -> Tuple[MAMLAgent, AStarPathfinder]:
        """
        Initialize all necessary components if they don't exist yet.
        
        Returns:
            Tuple[MAMLAgent, AStarPathfinder]: The initialized components
        """
        # Initialize A* Pathfinder if not already initialized
        if not self.pathfinder:
            # For now, initialize with empty Q-values - will be updated during training
            self.pathfinder = AStarPathfinder(
                q_values={},  # Will be populated during training
                config=self.config.pathfinder_config,
                metrics=PathfinderMetrics()
            )
            logger.info("Initialized A* Pathfinder")
        
        # Initialize Q-Learning agent if not already initialized
        if not self.q_learning_agent:
            # Create a Q-learning agent with the pathfinder
            self.q_learning_agent = QLearningAgent(
                config=AgentConfig(
                    agent_id="base_agent", 
                    q_learning_config=self.config.q_learning_config
                ),
                system_config=self.config.system_config,
                pathfinder=self.pathfinder
            )
            logger.info("Initialized Q-Learning agent with A* Pathfinding integration")
        
        # Initialize MAML agent if not already initialized
        if not self.maml_agent:
            # Create the MAML agent with the Q-Learning agent as base learner
            self.maml_agent = MAMLAgent(
                system_config=self.config.system_config,
                mcb_config=self.config.maml_config,
                base_agent=self.q_learning_agent
            )
            logger.info("Initialized MAML agent with Q-Learning as base learner")
        
        return self.maml_agent, self.pathfinder
    
    def _create_environment(self, 
                          obstacle_percentage: Optional[float] = None, 
                          maze_complexity: Optional[float] = None):
        """
        Create environment based on dataset type configuration.
        
        Args:
            obstacle_percentage (Optional[float]): Percentage of grid cells that are obstacles
            maze_complexity (Optional[float]): Complexity of the maze

        Returns:
            HierarchicalMultiAgentEnvironment: The created environment.
        """
        # Use provided values or defaults from config
        obstacle_pct = obstacle_percentage if obstacle_percentage is not None else self.config.grid_obstacles_percentage
        complexity = maze_complexity if maze_complexity is not None else self.config.maze_complexity
        
        # Get grid size from system config
        grid_size = self.config.system_config.grid_size if self.config.system_config else self.config.q_learning_config.state_space
        
        if self.config.dataset_type == DatasetType.GRID_WORLD:
            # Create a grid world environment with random obstacles
            import random
            import math
            
            # Calculate number of obstacles based on percentage
            total_cells = grid_size[0] * grid_size[1]
            num_obstacles = math.floor(total_cells * obstacle_pct)
            
            # Generate random obstacle positions (avoiding start and goal)
            obstacles = []
            start_pos = (0, 0)
            goal_pos = (grid_size[0]-1, grid_size[1]-1)
            
            while len(obstacles) < num_obstacles:
                pos = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
                if pos != start_pos and pos != goal_pos and pos not in obstacles:
                    obstacles.append(pos)
                    
            # Create environment
            env = HierarchicalMultiAgentEnvironment(grid_size=grid_size, obstacles=obstacles)
            logger.info(f"Created Grid World environment with size {grid_size} and {len(obstacles)} obstacles")
            return env
            
        elif self.config.dataset_type == DatasetType.MAZE:
            # Create a maze environment
            import random
            import numpy as np
            
            # Generate a maze using a simple algorithm
            width, height = grid_size
            maze = np.ones((height, width), dtype=int)  # 1 = wall, 0 = passage
            
            # Create a maze using randomized depth-first search
            def generate_maze(maze, x, y):
                maze[y][x] = 0
                
                # Define directions: right, down, left, up
                directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
                random.shuffle(directions)
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                        # Remove the wall between current and next cell
                        maze[y + dy//2][x + dx//2] = 0
                        generate_maze(maze, nx, ny)
            
            # Start from a random position
            start_x, start_y = 0, 0
            if width > 2 and height > 2:  # Ensure the maze is big enough
                generate_maze(maze, start_x, start_y)
            
            # Add some random passages based on complexity
            for y in range(height):
                for x in range(width):
                    if maze[y][x] == 1 and random.random() < (1 - complexity) * 0.3:
                        maze[y][x] = 0
            
            # Ensure start and goal are passages
            maze[0][0] = 0
            maze[height-1][width-1] = 0
            
            # Convert walls to obstacles for the environment
            obstacles = []
            for y in range(height):
                for x in range(width):
                    if maze[y][x] == 1:
                        obstacles.append((x, y))
            
            # Create environment
            env = HierarchicalMultiAgentEnvironment(grid_size=grid_size, obstacles=obstacles)
            logger.info(f"Created Maze environment with size {grid_size} and complexity {complexity}")
            return env
            
        elif self.config.dataset_type == DatasetType.CUSTOM:
            # For custom environments, user needs to provide it separately
            if self.custom_environment:
                return self.custom_environment
            else:
                logger.warning("Custom dataset type selected, but no environment provided. "
                              "Use set_custom_environment() to provide one.")
                return None
    
    def set_custom_environment(self, environment):
        """
        Set a custom environment for training/evaluation.
        
        Args:
            environment: The custom environment to use.
        """
        self.custom_environment = environment
        logger.info("Set custom environment")
    
    async def train(self, 
                   environment_id: Optional[str] = None,
                   environment: Optional[Any] = None, 
                   num_episodes: int = 1000,
                   checkpoint: bool = None) -> Dict[str, Any]:
        """
        Train the MetaQ-Star system on the specified environment.
        
        Args:
            environment_id (Optional[str]): ID of a previously registered environment
            environment (Optional[Any]): Environment to use for training (will be registered if provided)
            num_episodes (int): Number of training episodes
            checkpoint (bool): Whether to save checkpoints during training.
                If None, uses the value from config.
                
        Returns:
            Dict[str, Any]: Training results
            
        Note:
            You must either provide environment_id or environment, but not both.
        """
        # Input validation
        if environment_id is not None and environment is not None:
            raise ValueError("Provide either environment_id or environment, not both")
            
        if environment_id is None and environment is None:
            raise ValueError("Must provide either environment_id or environment")
        
        # Register the environment if provided directly
        if environment is not None:
            environment_id = f"env_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.environments[environment_id] = environment
        
        # Check if the environment exists
        if environment_id not in self.environments:
            raise ValueError(f"Environment with ID '{environment_id}' not found")
            
        # Ensure all components are initialized
        self._init_components()
        
        # Get environment
        environment = self.environments[environment_id]
        
        enable_checkpoints = checkpoint if checkpoint is not None else self.config.enable_checkpointing
        
        logger.info(f"Starting MetaQ-Star training for {num_episodes} episodes")
        
        # Check if we have task environments
        if not self.task_environments:
            raise ValueError("No task environments registered. Use register_task_environment() or register_task_environments_from_factory() first.")
        
        # Train the MAML agent with Q-Learning as the base learner
        results = await self.maml_agent.meta_train(
            task_environments=self.task_environments,
            num_meta_iterations=num_episodes
        )
        
        # Save checkpoint if enabled
        if enable_checkpoints:
            self.save_checkpoint()
            
        return results
    
    def _create_task_environments(self, num_tasks: int = 5) -> Dict[str, Any]:
        """
        Create multiple task environments for meta-learning.
        
        Args:
            num_tasks (int): Number of task environments to create.
            
        Returns:
            Dict[str, Any]: Dictionary mapping task IDs to environments.
        """
        task_environments = {}
        
        for i in range(num_tasks):
            task_id = f"task_{i}"
            
            # Create similar but different environments
            if self.config.dataset_type == DatasetType.GRID_WORLD:
                # Create grid worlds with varying obstacle densities
                obstacle_density = 0.1 + (i * 0.05)  # Vary obstacle density from 0.1 to 0.35
                env = self._create_environment(obstacle_percentage=obstacle_density)
                
            elif self.config.dataset_type == DatasetType.MAZE:
                # Create mazes with varying complexities
                complexity = 0.5 + (i * 0.1)  # Vary complexity from 0.5 to 0.9
                env = self._create_environment(maze_complexity=complexity)
                
            task_environments[task_id] = env
            
        logger.info(f"Created {num_tasks} task environments for meta-learning")
        return task_environments
    
    async def optimize(self, 
                      param_space: Dict[str, Dict], 
                      task_factory: Callable[[str], Any],
                      n_trials: int = 30, 
                      parallel_trials: int = 1,
                      distributed: bool = False,
                      num_tasks: int = 5,
                      num_test_tasks: int = 2) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the MetaQ-Star system using Optuna with optional Ray distribution.
        
        Args:
            param_space (Dict[str, Dict]): Parameter space definition for optimization
            task_factory (Callable[[str], Any]): Factory function that creates task environments 
            n_trials (int): Number of optimization trials.
            parallel_trials (int): Number of parallel trials.
            distributed (bool): Whether to use distributed optimization with Ray.
            num_tasks (int): Number of training tasks to create.
            num_test_tasks (int): Number of test tasks to create.
                
        Returns:
            Dict[str, Any]: Optimization results.
            
        Example param_space:
            {
                "maml_meta_lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "q_alpha": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
                "pathfinder_w1": {"type": "float", "low": 0.5, "high": 2.0}
            }
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        if not param_space:
            # Define the default combined parameter space for all components
            param_space = {
                # MAML parameters
                "maml_meta_lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "maml_inner_lr": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
                "maml_meta_batch_size": {"type": "int", "low": 2, "high": 10},
                "maml_adaptation_steps": {"type": "int", "low": 5, "high": 20},
                
                # Q-Learning parameters
                "q_alpha": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
                "q_gamma": {"type": "float", "low": 0.8, "high": 0.999},
                "q_epsilon": {"type": "float", "low": 0.5, "high": 1.0},
                "q_epsilon_decay": {"type": "float", "low": 0.9, "high": 0.999},
                
                # Pathfinder parameters
                "pathfinder_w1": {"type": "float", "low": 0.5, "high": 2.0},
                "pathfinder_w2": {"type": "float", "low": 0.5, "high": 2.0},
                "pathfinder_concurrent_expansions": {"type": "int", "low": 10, "high": 100},
            }
        
        # Define the objective function for optimization
        def objective_func(trial_params):
            """Objective function that Optuna/Ray will optimize."""
            # Create copies of our configurations to modify with trial parameters
            maml_config = copy.deepcopy(self.config.maml_config)
            q_learning_config = copy.deepcopy(self.config.q_learning_config)
            pathfinder_config = copy.deepcopy(self.config.pathfinder_config)
            system_config = copy.deepcopy(self.config.system_config)
            
            # Apply parameters from the trial
            for key, value in trial_params.items():
                if key.startswith("maml_"):
                    param_name = key[5:]  # Remove 'maml_' prefix
                    if hasattr(maml_config, param_name):
                        setattr(maml_config, param_name, value)
                elif key.startswith("q_"):
                    param_name = key[2:]  # Remove 'q_' prefix
                    if hasattr(q_learning_config, param_name):
                        setattr(q_learning_config, param_name, value)
                elif key.startswith("pathfinder_"):
                    param_name = key[11:]  # Remove 'pathfinder_' prefix
                    if hasattr(pathfinder_config, param_name):
                        setattr(pathfinder_config, param_name, value)
                elif key.startswith("system_"):
                    param_name = key[7:]  # Remove 'system_' prefix
                    if hasattr(system_config, param_name):
                        setattr(system_config, param_name, value)
            
            # Update agent configs in system_config with the modified q_learning_config
            for agent_config in system_config.agents_config:
                agent_config.q_learning_config = q_learning_config
            
            try:
                # Create the necessary components
                # 1. Create Pathfinder
                pathfinder = AStarPathfinder(q_values={}, config=pathfinder_config)
                
                # 2. Create Q-Learning Agent
                q_agent = QLearningAgent(
                    config=AgentConfig(agent_id="base_agent", q_learning_config=q_learning_config),
                    system_config=system_config,
                    pathfinder=pathfinder
                )
                
                # 3. Create MAML Agent
                maml_agent = MAMLAgent(
                    system_config=system_config,
                    mcb_config=maml_config,
                    base_agent=q_agent
                )
                
                # Create training task environments
                training_tasks = {f"task_{i}": task_factory(f"task_{i}") for i in range(num_tasks)}
                
                # Create testing task environments
                test_tasks = {f"test_task_{i}": task_factory(f"test_task_{i}") for i in range(num_test_tasks)}
                
                # Train the MAML agent
                asyncio.run(maml_agent.meta_train(
                    task_environments=training_tasks, 
                    num_meta_iterations=maml_config.meta_iterations
                ))
                
                # Evaluate on test tasks
                test_rewards = []
                for task_id, env in test_tasks.items():
                    reward = asyncio.run(maml_agent.adapt_to_new_task(
                        env, num_adaptation_steps=maml_config.adaptation_steps))
                    test_rewards.append(reward)
                
                avg_test_reward = sum(test_rewards) / len(test_rewards)
                return avg_test_reward
            except Exception as e:
                logger.error(f"Error during optimization trial: {e}")
                # Return a very low score for failed trials
                return float("-inf")
        
        # Optimize hyperparameters
        if distributed:
            # Make sure ray config is properly set for distributed optimization
            try:
                # Get Ray configuration from optimizer config
                from optimizer.optimizer_config import RayConfig, OptunaConfig
                ray_config = copy.deepcopy(self.config.optimizer_config.ray)
                optuna_config = OptunaConfig(n_trials=n_trials)
                
                # Use Ray with Optuna for distributed optimization
                results = self.optimizer.ray_distributor.integrate_with_optuna(
                    optuna_config=optuna_config,
                    objective_func=objective_func,
                    param_space=param_space,
                    n_samples=parallel_trials
                )
            except AttributeError as e:
                logger.error(f"Error setting up distributed optimization: {e}")
                logger.warning("Falling back to sequential optimization")
                distributed = False
        
        if not distributed:
            # Use standard Optuna optimization
            try:
                # First make sure we have a study initialized
                if not hasattr(self.optimizer.optuna_optimizer, 'study') or self.optimizer.optuna_optimizer.study is None:
                    self.optimizer.optuna_optimizer.create_study()
                
                # Create the objective function compatible with Optuna's API
                def optuna_objective(trial):
                    trial_params = {}
                    for param_name, param_spec in param_space.items():
                        if param_spec["type"] == "float":
                            trial_params[param_name] = trial.suggest_float(
                                param_name, 
                                param_spec["low"], 
                                param_spec["high"], 
                                log=param_spec.get("log", False)
                            )
                        elif param_spec["type"] == "int":
                            trial_params[param_name] = trial.suggest_int(
                                param_name, 
                                param_spec["low"], 
                                param_spec["high"], 
                                log=param_spec.get("log", False)
                            )
                        elif param_spec["type"] == "categorical":
                            trial_params[param_name] = trial.suggest_categorical(
                                param_name, 
                                param_spec["values"]
                            )
                    
                    return objective_func(trial_params)
                
                # Run the optimization
                self.optimizer.optuna_optimizer.study.optimize(
                    optuna_objective, n_trials=n_trials
                )
                
                # Get the best parameters
                results = {
                    'best_params': self.optimizer.optuna_optimizer.study.best_params,
                    'best_value': self.optimizer.optuna_optimizer.study.best_value
                }
            except Exception as e:
                logger.error(f"Error during optimization: {e}")
                results = {
                    'best_params': {},
                    'best_value': None,
                    'error': str(e)
                }
        
        # Update our configurations with the best parameters
        self._update_config_with_best_params(results.get('best_params', {}))
        
        logger.info(f"Hyperparameter optimization completed with best value: {results.get('best_value')}")
        return results
    
    def _update_config_with_best_params(self, best_params: Dict[str, Any]):
        """
        Update configuration with best parameters from optimization.
        
        Args:
            best_params (Dict[str, Any]): Best parameters from optimization.
        """
        if not best_params:
            logger.warning("No best parameters to update configuration with")
            return
            
        # Update MAML config
        for key, value in best_params.items():
            if key.startswith("maml_"):
                param_name = key[5:]  # Remove 'maml_' prefix
                if hasattr(self.config.maml_config, param_name):
                    setattr(self.config.maml_config, param_name, value)
                    logger.debug(f"Updated MAML parameter {param_name} to {value}")
        
        # Update Q-Learning config
        for key, value in best_params.items():
            if key.startswith("q_"):
                param_name = key[2:]  # Remove 'q_' prefix
                if hasattr(self.config.q_learning_config, param_name):
                    setattr(self.config.q_learning_config, param_name, value)
                    logger.debug(f"Updated Q-Learning parameter {param_name} to {value}")
        
        # Update Pathfinder config
        for key, value in best_params.items():
            if key.startswith("pathfinder_"):
                param_name = key[11:]  # Remove 'pathfinder_' prefix
                if hasattr(self.config.pathfinder_config, param_name):
                    setattr(self.config.pathfinder_config, param_name, value)
                    logger.debug(f"Updated Pathfinder parameter {param_name} to {value}")
        
        # Reinitialize components with updated configs
        self.maml_agent = None
        self.q_learning_agent = None
        self.pathfinder = None
    
    async def evaluate(self, 
                      environment_id: Optional[str] = None,
                      environment: Optional[Any] = None,
                      num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the MetaQ-Star system on the specified environment.
        
        Args:
            environment_id (Optional[str]): ID of a previously registered environment
            environment (Optional[Any]): Environment to use for evaluation (will be registered if provided)
            num_episodes (int): Number of evaluation episodes.
                
        Returns:
            Dict[str, Any]: Evaluation results.
            
        Note:
            You must either provide environment_id or environment, but not both.
        """
        # Input validation
        if environment_id is not None and environment is not None:
            raise ValueError("Provide either environment_id or environment, not both")
            
        if environment_id is None and environment is None:
            raise ValueError("Must provide either environment_id or environment")
        
        # Register the environment if provided directly
        if environment is not None:
            environment_id = f"env_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.environments[environment_id] = environment
        
        # Check if the environment exists
        if environment_id not in self.environments:
            raise ValueError(f"Environment with ID '{environment_id}' not found")
        
        # Initialize components if not already done
        self._init_components()
        
        # Get environment
        environment = self.environments[environment_id]
        
        logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        try:
            # Adapt the MAML agent to the new environment
            await self.maml_agent.adapt_to_new_task(environment)
            
            # Evaluate the performance
            results = await self.maml_agent._evaluate_on_task(environment, num_episodes)
            
            return results
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e), "success": False}
    
    def register_custom_model(self, model_type: str, model_class: Type, **kwargs):
        """
        Register a custom model to be used with the MetaQ-Star system.
        
        Args:
            model_type (str): Identifier for the model type
            model_class (Type): Class for the model
            **kwargs: Additional arguments to pass to the model constructor
        """
        # Create a registry for custom models if it doesn't exist
        if not hasattr(self, 'custom_models'):
            self.custom_models = {}
        
        self.custom_models[model_type] = {
            'class': model_class,
            'kwargs': kwargs
        }
        logger.info(f"Registered custom model type: {model_type}")
    
    def save_checkpoint(self, filepath: Optional[str] = None):
        """
        Save current model and configuration to a checkpoint file.
        
        Args:
            filepath (Optional[str]): Custom filepath for the checkpoint.
                If None, uses the default path and current timestamp.
        """
        if self.maml_agent is None:
            logger.warning("No model initialized, cannot save checkpoint.")
            return
        
        try:
            if filepath is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(
                    self.config.checkpoint_path, 
                    f"checkpoint_metaqstar_{timestamp}.pt"
                )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            checkpoint_data = {
                'config': self.config.dict(),
                'model_state': self.maml_agent.get_state_dict(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            import torch
            torch.save(checkpoint_data, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model and configuration from a checkpoint file.
        
        Args:
            filepath (str): Path to the checkpoint file.
        """
        try:
            import torch
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
            checkpoint_data = torch.load(filepath)
            
            # Recreate config from saved data
            self.config = MetaQStarConfig(**checkpoint_data['config'])
            
            # Reinitialize storage and components
            self.db_connectors = self._init_storage()
            self._init_components()
            
            # Load model state
            self.maml_agent.load_state_dict(checkpoint_data['model_state'])
            
            logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'optimizer') and self.optimizer:
                self.optimizer.shutdown()
            
            # Close database connections
            for connector in self.db_connectors.values():
                if hasattr(connector, 'close'):
                    connector.close()
            
            logger.info("Cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Example of using MetaQ-Star."""
    try:
        # Import required modules for environment creation
        from re_learning.q_learning import HierarchicalMultiAgentEnvironment
        import random
        import math
        
        # Create configuration
        config = MetaQStarConfig(
            storage_type=StorageType.LOCAL,
        )
        
        # Initialize system
        metaq = MetaQStar(config)
        
        # Define a task factory function to create environments for training
        def create_environment(task_id):
            # Extract a seed from the task_id
            seed = int(task_id.split('_')[-1]) if '_' in task_id else 0
            random.seed(seed)
            
            # Create a grid world environment with random obstacles
            grid_size = (10, 10)
            obstacle_pct = 0.1 + (seed * 0.05) % 0.3
            
            # Calculate number of obstacles based on percentage
            total_cells = grid_size[0] * grid_size[1]
            num_obstacles = math.floor(total_cells * obstacle_pct)
            
            # Generate random obstacle positions (avoiding start and goal)
            obstacles = []
            start_pos = (0, 0)
            goal_pos = (grid_size[0]-1, grid_size[1]-1)
            
            while len(obstacles) < num_obstacles:
                pos = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
                if pos != start_pos and pos != goal_pos and pos not in obstacles:
                    obstacles.append(pos)
            
            # Create and return the environment
            return HierarchicalMultiAgentEnvironment(grid_size=grid_size, obstacles=obstacles)
        
        # Register task environments
        metaq.register_task_environments_from_factory(create_environment, num_tasks=5)
        
        # Create an environment for training/evaluation
        train_env = create_environment("train")
        metaq.environments["train_env"] = train_env
        
        # Optional: Optimize hyperparameters
        optimize_results = await metaq.optimize(
            param_space={
                "maml_meta_lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "q_alpha": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
            },
            task_factory=create_environment,
            n_trials=5,  # Reduced for demonstration
            parallel_trials=1,
            distributed=False
        )
        print("Optimization results:", optimize_results)
        
        # Train the model
        results = await metaq.train(
            environment_id="train_env",
            num_episodes=30  # Reduced for demonstration
        )
        print("Training results:", results)
        
        # Evaluate the model
        eval_results = await metaq.evaluate(
            environment_id="train_env",
            num_episodes=5
        )
        print("Evaluation results:", eval_results)
        
        # Save checkpoint
        metaq.save_checkpoint()
        
        # Clean up
        metaq.cleanup()
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
