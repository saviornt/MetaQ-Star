"""
Ray Distributed Computing Module for Q-Learning and MAML.

This module provides utilities to scale training and hyperparameter optimization
across multiple cores and machines using Ray. It integrates with the Optuna
hyperparameter optimization for distributed parameter tuning.
"""

import logging
import os
import pickle
import time
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import asyncio
import copy

# Import Ray conditionally to handle environments where it's not installed
try:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    logging.warning("Ray not installed. Distributed optimization unavailable.")

from .optimizer_config import RayConfig, OptunaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RayDistributor:
    """
    Ray-based distributed computing for scaling Q-Learning and MAML.
    
    This class provides methods to distribute workloads across multiple
    cores/machines using Ray. It can be used for both distributed training
    and hyperparameter optimization in conjunction with Optuna.
    
    Attributes:
        config (RayConfig): Configuration for Ray distributed computing.
        is_initialized (bool): Whether Ray has been initialized.
        ray_resources (Dict): Resources available to Ray.
    """
    
    def __init__(self, config: RayConfig):
        """
        Initialize the RayDistributor.
        
        Args:
            config (RayConfig): Configuration settings for Ray.
        """
        self.config = config
        self.is_initialized = False
        self.ray_resources = {}
        
        # Initialize Ray if available
        if HAS_RAY:
            self.initialize_ray()
        else:
            logger.warning("Ray not available. Using local computation only.")
    
    def initialize_ray(self):
        """Initialize Ray with the configured settings."""
        if not HAS_RAY:
            logger.warning("Ray not installed. Skipping initialization.")
            return
        
        if ray.is_initialized():
            logger.info("Ray already initialized. Skipping initialization.")
            self.is_initialized = True
            self.ray_resources = ray.cluster_resources()
            return
        
        try:
            # Initialize Ray with given configuration
            ray.init(
                address=self.config.address,
                num_cpus=self.config.num_cpus,
                num_gpus=self.config.num_gpus,
                _memory=self.config.memory,
                object_store_memory=self.config.object_store_memory,
                _redis_max_memory=self.config.redis_max_memory
            )
            
            self.is_initialized = True
            self.ray_resources = ray.cluster_resources()
            logger.info(f"Ray initialized with resources: {self.ray_resources}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            logger.info("Falling back to local computation")
    
    def shutdown(self):
        """Shutdown Ray if it is initialized."""
        if HAS_RAY and self.is_initialized:
            ray.shutdown()
            self.is_initialized = False
            logger.info("Ray shutdown completed")
    
    @ray.remote
    def _remote_train_agent(agent_config, env_creator, episodes, eval_episodes=5):
        """
        Remote function for training a Q-learning agent.
        
        Note: This is defined as a static method to work with Ray.
        
        Args:
            agent_config: Configuration for the agent.
            env_creator: Function that creates an environment.
            episodes: Number of training episodes.
            eval_episodes: Number of episodes for evaluation.
            
        Returns:
            Dict: Results including reward and trained agent.
        """
        from re_learning.q_learning import HierarchicalAgentSystem
        import asyncio
        
        # Create environment
        env = env_creator()
        
        # Create agent system
        agent_system = HierarchicalAgentSystem(agent_config)
        
        # Train the agent using asyncio.run
        asyncio.run(agent_system.train(env, episodes))
        
        # Evaluate the trained agent
        total_reward = 0
        for _ in range(eval_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = asyncio.run(agent_system.select_action(state))
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
            total_reward += episode_reward
        
        avg_reward = total_reward / eval_episodes
        
        # Get serializable state of the agent
        agent_state = agent_system.get_serializable_state()
        
        return {
            "reward": avg_reward,
            "agent_state": agent_state
        }
    
    def distribute_q_learning(self, 
                              env_creator: Callable,
                              base_config: Any,
                              param_variations: List[Dict[str, Any]],
                              episodes: int = 100,
                              eval_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        Distribute Q-learning training across multiple workers.
        
        Args:
            env_creator: Function that creates environments.
            base_config: Base configuration for agents.
            param_variations: List of parameter variations to try.
            episodes: Number of episodes for training.
            eval_episodes: Number of episodes for evaluation.
            
        Returns:
            List[Dict]: Results from all distributed training runs.
        """
        if not HAS_RAY or not self.is_initialized:
            logger.warning("Ray not initialized. Running in sequential mode.")
            return self._sequential_q_learning(
                env_creator, base_config, param_variations, episodes, eval_episodes)
        
        # Create tasks for all parameter variations
        tasks = []
        for params in param_variations:
            # Create a copy of the base config and update with the parameter variation
            config_copy = copy.deepcopy(base_config)
            
            # Apply parameter variations (this depends on the structure of your config)
            for key, value in params.items():
                # Handle nested parameters
                if '.' in key:
                    parts = key.split('.')
                    obj = config_copy
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            logger.warning(f"Config does not have attribute {part}")
                            break
                    else:
                        setattr(obj, parts[-1], value)
                else:
                    if hasattr(config_copy, key):
                        setattr(config_copy, key, value)
                    else:
                        logger.warning(f"Config does not have attribute {key}")
            
            # Create remote task
            task = self._remote_train_agent.remote(
                config_copy, env_creator, episodes, eval_episodes)
            tasks.append((task, params))
        
        # Wait for all tasks to complete
        results = []
        for task, params in tasks:
            result = ray.get(task)
            result["params"] = params
            results.append(result)
            
        logger.info(f"Completed {len(results)} distributed training runs")
        return results
    
    def _sequential_q_learning(self, 
                              env_creator: Callable,
                              base_config: Any,
                              param_variations: List[Dict[str, Any]],
                              episodes: int = 100,
                              eval_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        Run Q-learning sequentially (fallback when Ray is not available).
        
        Args:
            env_creator: Function that creates environments.
            base_config: Base configuration for agents.
            param_variations: List of parameter variations to try.
            episodes: Number of episodes for training.
            eval_episodes: Number of episodes for evaluation.
            
        Returns:
            List[Dict]: Results from all training runs.
        """
        from re_learning.q_learning import HierarchicalAgentSystem
        
        results = []
        for params in param_variations:
            # Create a copy of the base config and update with the parameter variation
            config_copy = copy.deepcopy(base_config)
            
            # Apply parameter variations (this depends on the structure of your config)
            for key, value in params.items():
                # Handle nested parameters
                if '.' in key:
                    parts = key.split('.')
                    obj = config_copy
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            logger.warning(f"Config does not have attribute {part}")
                            break
                    else:
                        setattr(obj, parts[-1], value)
                else:
                    if hasattr(config_copy, key):
                        setattr(config_copy, key, value)
                    else:
                        logger.warning(f"Config does not have attribute {key}")
            
            # Create environment
            env = env_creator()
            
            # Create agent system
            agent_system = HierarchicalAgentSystem(config_copy)
            
            # Train the agent
            asyncio.run(agent_system.train(env, episodes))
            
            # Evaluate the trained agent
            total_reward = 0
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = asyncio.run(agent_system.select_action(state))
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                total_reward += episode_reward
            
            avg_reward = total_reward / eval_episodes
            
            # Get serializable state of the agent
            agent_state = agent_system.get_serializable_state()
            
            results.append({
                "reward": avg_reward,
                "agent_state": agent_state,
                "params": params
            })
            
            logger.info(f"Completed run with params {params}, reward: {avg_reward}")
        
        return results
    
    @ray.remote
    def _remote_meta_train_agent(maml_config, system_config, task_creators, test_task_creators, 
                                num_meta_iterations=10, num_adaptation_steps=5):
        """
        Remote function for meta-training a MAML agent.
        
        Args:
            maml_config: Configuration for MAML.
            system_config: Configuration for the agent system.
            task_creators: List of functions that create training tasks.
            test_task_creators: List of functions that create test tasks.
            num_meta_iterations: Number of meta-training iterations.
            num_adaptation_steps: Number of adaptation steps for evaluation.
            
        Returns:
            Dict: Results including reward and trained agent.
        """
        from src.meta_learning.maml import MAMLAgent
        import asyncio
        
        # Create training and testing environments
        training_tasks = {f"task_{i}": creator() for i, creator in enumerate(task_creators)}
        test_tasks = {f"test_task_{i}": creator() for i, creator in enumerate(test_task_creators)}
        
        # Create MAML agent
        maml_agent = MAMLAgent(system_config, maml_config)
        
        # Meta-train the agent
        asyncio.run(maml_agent.meta_train(
            training_tasks, num_meta_iterations=num_meta_iterations))
        
        # Evaluate on test tasks
        test_rewards = []
        for task_id, env in test_tasks.items():
            reward = asyncio.run(maml_agent.adapt_to_new_task(
                env, num_adaptation_steps=num_adaptation_steps))
            test_rewards.append(reward)
        
        avg_test_reward = sum(test_rewards) / len(test_rewards)
        
        # Get serializable state of the agent
        agent_state = maml_agent.get_serializable_state()
        
        return {
            "reward": avg_test_reward,
            "agent_state": agent_state
        }
    
    def distribute_maml(self,
                        task_creators: List[Callable],
                        test_task_creators: List[Callable],
                        base_maml_config: Any,
                        base_system_config: Any,
                        param_variations: List[Dict[str, Any]],
                        num_meta_iterations: int = 10,
                        num_adaptation_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Distribute MAML training across multiple workers.
        
        Args:
            task_creators: List of functions that create training tasks.
            test_task_creators: List of functions that create test tasks.
            base_maml_config: Base configuration for MAML.
            base_system_config: Base configuration for agent systems.
            param_variations: List of parameter variations to try.
            num_meta_iterations: Number of meta-training iterations.
            num_adaptation_steps: Number of adaptation steps for evaluation.
            
        Returns:
            List[Dict]: Results from all distributed training runs.
        """
        if not HAS_RAY or not self.is_initialized:
            logger.warning("Ray not initialized. Running in sequential mode.")
            return self._sequential_maml(
                task_creators, test_task_creators, base_maml_config, base_system_config,
                param_variations, num_meta_iterations, num_adaptation_steps)
        
        # Create tasks for all parameter variations
        tasks = []
        for params in param_variations:
            # Create copies of the base configs
            maml_config_copy = copy.deepcopy(base_maml_config)
            system_config_copy = copy.deepcopy(base_system_config)
            
            # Apply parameter variations
            for key, value in params.items():
                if key.startswith("maml_"):
                    # Set MAML-specific parameters
                    maml_param_name = key[len("maml_"):]
                    if hasattr(maml_config_copy, maml_param_name):
                        setattr(maml_config_copy, maml_param_name, value)
                else:
                    # Set system-level parameters
                    if hasattr(system_config_copy, key):
                        setattr(system_config_copy, key, value)
            
            # Create remote task
            task = self._remote_meta_train_agent.remote(
                maml_config_copy, system_config_copy, task_creators, test_task_creators,
                num_meta_iterations, num_adaptation_steps)
            tasks.append((task, params))
        
        # Wait for all tasks to complete
        results = []
        for task, params in tasks:
            result = ray.get(task)
            result["params"] = params
            results.append(result)
            
        logger.info(f"Completed {len(results)} distributed MAML training runs")
        return results
    
    def _sequential_maml(self,
                        task_creators: List[Callable],
                        test_task_creators: List[Callable],
                        base_maml_config: Any,
                        base_system_config: Any,
                        param_variations: List[Dict[str, Any]],
                        num_meta_iterations: int = 10,
                        num_adaptation_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Run MAML sequentially (fallback when Ray is not available).
        
        Args:
            task_creators: List of functions that create training tasks.
            test_task_creators: List of functions that create test tasks.
            base_maml_config: Base configuration for MAML.
            base_system_config: Base configuration for agent systems.
            param_variations: List of parameter variations to try.
            num_meta_iterations: Number of meta-training iterations.
            num_adaptation_steps: Number of adaptation steps for evaluation.
            
        Returns:
            List[Dict]: Results from all training runs.
        """
        from src.meta_learning.maml import MAMLAgent
        
        results = []
        for params in param_variations:
            # Create copies of the base configs
            maml_config_copy = copy.deepcopy(base_maml_config)
            system_config_copy = copy.deepcopy(base_system_config)
            
            # Apply parameter variations
            for key, value in params.items():
                if key.startswith("maml_"):
                    # Set MAML-specific parameters
                    maml_param_name = key[len("maml_"):]
                    if hasattr(maml_config_copy, maml_param_name):
                        setattr(maml_config_copy, maml_param_name, value)
                else:
                    # Set system-level parameters
                    if hasattr(system_config_copy, key):
                        setattr(system_config_copy, key, value)
            
            # Create training and testing environments
            training_tasks = {f"task_{i}": creator() for i, creator in enumerate(task_creators)}
            test_tasks = {f"test_task_{i}": creator() for i, creator in enumerate(test_task_creators)}
            
            # Create MAML agent
            maml_agent = MAMLAgent(system_config_copy, maml_config_copy)
            
            # Meta-train the agent
            asyncio.run(maml_agent.meta_train(
                training_tasks, num_meta_iterations=num_meta_iterations))
            
            # Evaluate on test tasks
            test_rewards = []
            for task_id, env in test_tasks.items():
                reward = asyncio.run(maml_agent.adapt_to_new_task(
                    env, num_adaptation_steps=num_adaptation_steps))
                test_rewards.append(reward)
            
            avg_test_reward = sum(test_rewards) / len(test_rewards)
            
            # Get serializable state of the agent
            agent_state = maml_agent.get_serializable_state()
            
            results.append({
                "reward": avg_test_reward,
                "agent_state": agent_state,
                "params": params
            })
            
            logger.info(f"Completed run with params {params}, reward: {avg_test_reward}")
        
        return results
    
    def integrate_with_optuna(self, 
                             optuna_config: OptunaConfig, 
                             objective_func: Callable, 
                             param_space: Dict[str, Any],
                             n_samples: int = 10) -> Dict[str, Any]:
        """
        Integrate Ray with Optuna for distributed hyperparameter optimization.
        
        Args:
            optuna_config: Configuration for Optuna.
            objective_func: Function to optimize.
            param_space: Dictionary defining the hyperparameter search space.
            n_samples: Number of samples for Ray Tune.
            
        Returns:
            Dict: Best hyperparameters found.
        """
        if not HAS_RAY or not self.is_initialized:
            logger.warning("Ray not initialized. Cannot integrate with Optuna in distributed mode.")
            return {}
        
        # Convert Optuna param space to Ray Tune format
        tune_param_space = {}
        for param_name, param_spec in param_space.items():
            if isinstance(param_spec, dict):
                if param_spec.get("type") == "categorical":
                    tune_param_space[param_name] = tune.choice(param_spec["values"])
                elif param_spec.get("type") == "float":
                    if param_spec.get("log", False):
                        tune_param_space[param_name] = tune.loguniform(
                            param_spec["low"], param_spec["high"])
                    else:
                        tune_param_space[param_name] = tune.uniform(
                            param_spec["low"], param_spec["high"])
                elif param_spec.get("type") == "int":
                    if param_spec.get("log", False):
                        tune_param_space[param_name] = tune.lograndint(
                            param_spec["low"], param_spec["high"])
                    else:
                        tune_param_space[param_name] = tune.randint(
                            param_spec["low"], param_spec["high"])
        
        # Create Optuna search algorithm for Ray Tune
        optuna_search = OptunaSearch(
            space=tune_param_space,
            metric="reward",
            mode=optuna_config.direction,
            points_to_evaluate=None
        )
        
        # Define Ray Tune scheduler
        scheduler = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=2
        )
        
        # Define the Ray Tune trainable function
        def trainable(config):
            result = objective_func(config)
            tune.report(reward=result)
        
        # Run the Ray Tune study
        analysis = tune.run(
            trainable,
            name=optuna_config.study_name,
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=n_samples,
            resources_per_trial={"cpu": 1, "gpu": self.config.num_gpus / n_samples if self.config.num_gpus else 0},
            verbose=1
        )
        
        # Get the best hyperparameters
        best_params = analysis.best_config
        
        logger.info(f"Completed distributed optimization with Optuna and Ray Tune")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def distribute_neural_network_training(self,
                                          model_factory: Callable,
                                          dataset: Any,
                                          train_func: Callable,
                                          eval_func: Callable,
                                          num_workers: int = 2,
                                          batch_size: int = 32,
                                          epochs: int = 10) -> Any:
        """
        Distribute neural network training across multiple workers.
        
        Args:
            model_factory: Function that creates model instances.
            dataset: Dataset to use for training and evaluation.
            train_func: Function for training a model.
            eval_func: Function for evaluating a model.
            num_workers: Number of workers to use.
            batch_size: Batch size for training.
            epochs: Number of epochs for training.
            
        Returns:
            Any: Trained model.
        """
        if not HAS_RAY or not self.is_initialized:
            logger.warning("Ray not initialized. Running in single process mode.")
            model = model_factory()
            train_func(model, dataset, batch_size, epochs)
            metrics = eval_func(model, dataset)
            return model, metrics
        
        # Define remote training function
        @ray.remote
        def remote_train_and_eval(model_config, dataset_ref, epoch_subset):
            # Create model
            model = model_factory(**model_config)
            
            # Get dataset from object store
            dataset = ray.get(dataset_ref)
            
            # Train the model for a subset of epochs
            train_func(model, dataset, batch_size, len(epoch_subset))
            
            return model.get_serializable_state()
        
        # Parallelize training across different epoch ranges
        epoch_splits = []
        epochs_per_worker = max(1, epochs // num_workers)
        for i in range(0, epochs, epochs_per_worker):
            epoch_splits.append(list(range(i, min(i + epochs_per_worker, epochs))))
        
        # Put dataset in Ray object store (if possible)
        try:
            dataset_ref = ray.put(dataset)
        except Exception:
            # If dataset can't be put in object store (e.g., too large),
            # fall back to single process
            logger.warning("Could not put dataset in Ray object store. Running in single process mode.")
            model = model_factory()
            train_func(model, dataset, batch_size, epochs)
            metrics = eval_func(model, dataset)
            return model, metrics
        
        # Create empty model configuration - assuming model_factory doesn't need params
        model_config = {}
        
        # Launch remote training tasks
        tasks = []
        for epoch_subset in epoch_splits:
            task = remote_train_and_eval.remote(model_config, dataset_ref, epoch_subset)
            tasks.append(task)
        
        # Get model states from all workers
        model_states = ray.get(tasks)
        
        # Create final model
        final_model = model_factory()
        
        # Combine model states (implementation depends on your model structure)
        # For example, averaging weights for neural networks
        final_model.load_combined_state(model_states)
        
        # Evaluate final model
        metrics = eval_func(final_model, dataset)
        
        return final_model, metrics
    
    def pbt_optimize(self,
                    model_factory: Callable,
                    env_creator: Callable,
                    param_space: Dict[str, Any],
                    train_func: Callable,
                    eval_func: Callable,
                    population_size: int = 4,
                    num_iterations: int = 10) -> Dict[str, Any]:
        """
        Use Population Based Training (PBT) for optimization.
        
        Args:
            model_factory: Function that creates model instances.
            env_creator: Function that creates environment instances.
            param_space: Dictionary defining the hyperparameter search space.
            train_func: Function for training a model.
            eval_func: Function for evaluating a model.
            population_size: Size of the population.
            num_iterations: Number of iterations.
            
        Returns:
            Dict: Best parameters found.
        """
        if not HAS_RAY or not self.is_initialized:
            logger.warning("Ray not initialized. Cannot use PBT.")
            return {}
        
        # Convert parameter space to Ray Tune format
        tune_param_space = {}
        for param_name, param_spec in param_space.items():
            if isinstance(param_spec, dict):
                if param_spec.get("type") == "categorical":
                    tune_param_space[param_name] = tune.choice(param_spec["values"])
                elif param_spec.get("type") == "float":
                    if param_spec.get("log", False):
                        tune_param_space[param_name] = tune.loguniform(
                            param_spec["low"], param_spec["high"])
                    else:
                        tune_param_space[param_name] = tune.uniform(
                            param_spec["low"], param_spec["high"])
                elif param_spec.get("type") == "int":
                    if param_spec.get("log", False):
                        tune_param_space[param_name] = tune.lograndint(
                            param_spec["low"], param_spec["high"])
                    else:
                        tune_param_space[param_name] = tune.randint(
                            param_spec["low"], param_spec["high"])
        
        # Create PBT scheduler
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="reward",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations=tune_param_space
        )
        
        # Define the Ray Tune trainable function
        def trainable(config, checkpoint_dir=None):
            # Create model
            model = model_factory(**config)
            
            # Load checkpoint if available
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
                model.load_state(checkpoint_path)
            
            # Create environment
            env = env_creator()
            
            # Train for one iteration
            train_func(model, env, 1)
            
            # Evaluate
            reward = eval_func(model, env)
            
            # Create a checkpoint
            checkpoint_path = os.path.join(tune.get_trial_dir(), "checkpoint")
            model.save_state(checkpoint_path)
            
            # Report results
            tune.report(reward=reward)
        
        # Run PBT
        analysis = tune.run(
            trainable,
            name="pbt_optimization",
            scheduler=pbt_scheduler,
            num_samples=population_size,
            stop={"training_iteration": num_iterations},
            resources_per_trial={"cpu": 1, "gpu": self.config.num_gpus / population_size if self.config.num_gpus else 0},
            keep_checkpoints_num=1,
            checkpoint_score_attr="reward",
            verbose=1
        )
        
        # Get the best hyperparameters
        best_params = analysis.best_config
        
        logger.info(f"Completed PBT optimization")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
