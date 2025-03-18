"""
Unified Optimizer Module for Q-Learning and MAML.

This module provides a unified interface combining Optuna's hyperparameter
optimization capabilities with Ray's distributed computing features,
enabling efficient optimization of Q-Learning, MAML, and other models.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import asyncio
import copy
import os
import pickle
import datetime

from .optuna import OptunaOptimizer, get_q_learning_param_space, get_maml_param_space
from .ray import RayDistributor
from .optimizer_config import OptimizerConfig, RayConfig, OptunaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Optimizer:
    """
    Unified optimizer combining Optuna hyperparameter optimization and Ray distributed computing.
    
    This class provides a simplified interface to optimize hyperparameters and
    distribute training workloads for Q-Learning, MAML, and other models.
    
    Attributes:
        config (OptimizerConfig): Configuration for both Optuna and Ray.
        optuna_optimizer (OptunaOptimizer): Optuna-based optimizer.
        ray_distributor (RayDistributor): Ray-based distributed computing.
        best_params (Dict): Best hyperparameters found during optimization.
        optimization_history (List): History of optimization trials.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize the unified optimizer.
        
        Args:
            config (Optional[OptimizerConfig]): Configuration settings.
                If None, default configurations will be used.
        """
        self.config = config or OptimizerConfig()
        
        # Initialize Optuna optimizer
        self.optuna_optimizer = OptunaOptimizer(self.config)
        
        # Initialize Ray distributor
        self.ray_distributor = RayDistributor(self.config.ray)
        
        # Store optimization results
        self.best_params = None
        self.optimization_history = []
        
        logger.info("Unified optimizer initialized")
    
    def optimize_q_learning(self,
                           environment_factory: Callable,
                           base_agent_config: Any,
                           system_config: Any,
                           param_space: Optional[Dict[str, Any]] = None,
                           n_trials: int = 50,
                           parallel_trials: int = 1,
                           eval_episodes: int = 10,
                           distributed: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Q-Learning agents.
        
        Args:
            environment_factory: Function that creates environment instances.
            base_agent_config: Base agent configuration to modify with trial parameters.
            system_config: System configuration for hierarchical agents.
            param_space: Dictionary defining hyperparameter search space.
                If None, a default space will be used.
            n_trials: Number of optimization trials to run.
            parallel_trials: Number of trials to run in parallel (when distributed=True).
            eval_episodes: Number of episodes for evaluation.
            distributed: Whether to use distributed optimization with Ray.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        # Use default parameter space if none provided
        if param_space is None:
            param_space = get_q_learning_param_space()
            logger.info("Using default Q-Learning hyperparameter space")
        
        if distributed and self.ray_distributor.is_initialized:
            logger.info("Using Ray for distributed Q-Learning optimization")
            
            # Create function to optimize using Optuna
            def objective_func(config):
                # Create agent config from parameters
                agent_config = copy.deepcopy(base_agent_config)
                
                # Apply parameters
                for key, value in config.items():
                    if key.startswith("q_learning_"):
                        # Set Q-learning specific parameters
                        q_param_name = key[len("q_learning_"):]
                        if hasattr(agent_config.q_learning_config, q_param_name):
                            setattr(agent_config.q_learning_config, q_param_name, value)
                    else:
                        # Set agent-level parameters
                        if hasattr(agent_config, key):
                            setattr(agent_config, key, value)
                
                # Update system config with the modified agent config
                system_config_copy = copy.deepcopy(system_config)
                system_config_copy.agents_config = [agent_config]
                
                # Create environment
                env = environment_factory()
                
                # Create and train the agent system
                from re_learning.q_learning import HierarchicalAgentSystem
                agent_system = HierarchicalAgentSystem(system_config_copy)
                
                # Run training
                asyncio.run(agent_system.train(env, agent_config.q_learning_config.episodes))
                
                # Evaluate
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
                
                return total_reward / eval_episodes
            
            # Create Optuna configuration for this run
            optuna_config = copy.deepcopy(self.config.optuna)
            optuna_config.n_trials = n_trials
            
            # Use Ray with Optuna for distributed optimization
            best_params = self.ray_distributor.integrate_with_optuna(
                optuna_config=optuna_config,
                objective_func=objective_func,
                param_space=param_space,
                n_samples=parallel_trials
            )
            
            self.best_params = best_params
            
        else:
            # Use standard Optuna optimization
            logger.info("Using Optuna for Q-Learning optimization")
            
            # Create or get the Optuna study
            if self.optuna_optimizer.study is None:
                self.optuna_optimizer.create_study()
            
            # Run optimization
            best_params = self.optuna_optimizer.optimize_q_learning(
                environment_factory=environment_factory,
                param_space=param_space,
                base_agent_config=base_agent_config,
                system_config=system_config,
                eval_episodes=eval_episodes,
                n_trials=n_trials
            )
            
            self.best_params = best_params
            self.optimization_history = self.optuna_optimizer.optimization_history
        
        logger.info(f"Q-Learning optimization completed. Best params: {self.best_params}")
        return self.best_params
    
    def optimize_maml(self,
                     task_factory: Callable,
                     base_system_config: Any,
                     base_maml_config: Any,
                     param_space: Optional[Dict[str, Any]] = None,
                     n_trials: int = 30,
                     parallel_trials: int = 1,
                     num_tasks: int = 5,
                     num_test_tasks: int = 2,
                     distributed: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters for MAML (Model-Agnostic Meta-Learning).
        
        Args:
            task_factory: Function that creates task environments.
            base_system_config: Base system configuration.
            base_maml_config: Base MAML configuration.
            param_space: Dictionary defining hyperparameter search space.
                If None, a default space will be used.
            n_trials: Number of optimization trials to run.
            parallel_trials: Number of trials to run in parallel (when distributed=True).
            num_tasks: Number of training tasks to use.
            num_test_tasks: Number of test tasks to use.
            distributed: Whether to use distributed optimization with Ray.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        # Use default parameter space if none provided
        if param_space is None:
            param_space = get_maml_param_space()
            logger.info("Using default MAML hyperparameter space")
        
        if distributed and self.ray_distributor.is_initialized:
            logger.info("Using Ray for distributed MAML optimization")
            
            # Create function to optimize using Optuna
            def objective_func(config):
                # Create copies of the base configs
                maml_config_copy = copy.deepcopy(base_maml_config)
                system_config_copy = copy.deepcopy(base_system_config)
                
                # Apply parameter variations
                for key, value in config.items():
                    if key.startswith("maml_"):
                        # Set MAML-specific parameters
                        maml_param_name = key[len("maml_"):]
                        if hasattr(maml_config_copy, maml_param_name):
                            setattr(maml_config_copy, maml_param_name, value)
                    elif key.startswith("q_learning_"):
                        # Set Q-learning parameters in agent configs
                        q_param_name = key[len("q_learning_"):]
                        for agent_config in system_config_copy.agents_config:
                            if hasattr(agent_config.q_learning_config, q_param_name):
                                setattr(agent_config.q_learning_config, q_param_name, value)
                    else:
                        # Set system-level parameters
                        if hasattr(system_config_copy, key):
                            setattr(system_config_copy, key, value)
                
                # Create training and testing tasks
                training_tasks = {f"task_{i}": task_factory(i) 
                                 for i in range(num_tasks)}
                
                test_tasks = {f"test_task_{i}": task_factory(num_tasks + i) 
                             for i in range(num_test_tasks)}
                
                # Create and train MAML agent
                from src.meta_learning.maml import MAMLAgent
                maml_agent = MAMLAgent(system_config_copy, maml_config_copy)
                
                # Run meta-training
                asyncio.run(maml_agent.meta_train(
                    training_tasks, num_meta_iterations=maml_config_copy.meta_iterations))
                
                # Evaluate on test tasks
                test_rewards = []
                for task_id, env in test_tasks.items():
                    reward = asyncio.run(maml_agent.adapt_to_new_task(
                        env, num_adaptation_steps=maml_config_copy.adaptation_steps))
                    test_rewards.append(reward)
                
                avg_test_reward = sum(test_rewards) / len(test_rewards)
                return avg_test_reward
            
            # Create Optuna configuration for this run
            optuna_config = copy.deepcopy(self.config.optuna)
            optuna_config.n_trials = n_trials
            
            # Use Ray with Optuna for distributed optimization
            best_params = self.ray_distributor.integrate_with_optuna(
                optuna_config=optuna_config,
                objective_func=objective_func,
                param_space=param_space,
                n_samples=parallel_trials
            )
            
            self.best_params = best_params
            
        else:
            # Use standard Optuna optimization
            logger.info("Using Optuna for MAML optimization")
            
            # Create or get the Optuna study
            if self.optuna_optimizer.study is None:
                self.optuna_optimizer.create_study()
            
            # Define task factory wrapper
            def task_environment_factory(task_id):
                return task_factory(task_id)
            
            # Run optimization
            best_params = self.optuna_optimizer.optimize_maml(
                task_environment_factory=task_environment_factory,
                param_space=param_space,
                base_system_config=base_system_config,
                base_maml_config=base_maml_config,
                num_tasks=num_tasks,
                num_test_tasks=num_test_tasks,
                n_trials=n_trials
            )
            
            self.best_params = best_params
            self.optimization_history = self.optuna_optimizer.optimization_history
        
        logger.info(f"MAML optimization completed. Best params: {self.best_params}")
        return self.best_params
    
    def optimize_pathfinder(self,
                           environment_factory: Callable,
                           base_config: Any,
                           param_space: Dict[str, Any],
                           eval_func: Callable,
                           n_trials: int = 30,
                           parallel_trials: int = 1,
                           distributed: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Pathfinder models.
        
        Args:
            environment_factory: Function that creates environment instances.
            base_config: Base configuration for Pathfinder.
            param_space: Dictionary defining hyperparameter search space.
            eval_func: Function to evaluate Pathfinder performance.
            n_trials: Number of optimization trials to run.
            parallel_trials: Number of trials to run in parallel (when distributed=True).
            distributed: Whether to use distributed optimization with Ray.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        if distributed and self.ray_distributor.is_initialized:
            logger.info("Using Ray for distributed Pathfinder optimization")
            
            # Create function to optimize using Optuna
            def objective_func(config):
                # Create a copy of the base config and update with the parameter variation
                config_copy = copy.deepcopy(base_config)
                
                # Apply parameter variations
                for key, value in config.items():
                    if hasattr(config_copy, key):
                        setattr(config_copy, key, value)
                
                # Create environment
                env = environment_factory()
                
                # Evaluate with the provided evaluation function
                score = eval_func(config_copy, env)
                return score
            
            # Create Optuna configuration for this run
            optuna_config = copy.deepcopy(self.config.optuna)
            optuna_config.n_trials = n_trials
            
            # Use Ray with Optuna for distributed optimization
            best_params = self.ray_distributor.integrate_with_optuna(
                optuna_config=optuna_config,
                objective_func=objective_func,
                param_space=param_space,
                n_samples=parallel_trials
            )
            
            self.best_params = best_params
            
        else:
            # Use standard Optuna optimization
            logger.info("Using Optuna for Pathfinder optimization")
            
            # Create or get the Optuna study
            if self.optuna_optimizer.study is None:
                self.optuna_optimizer.create_study()
            
            # Define the objective function for Pathfinder optimization
            def objective(trial):
                # Sample hyperparameters from the search space
                trial_params = {}
                for param_name, param_spec in param_space.items():
                    if isinstance(param_spec, dict):
                        if param_spec.get("type") == "categorical":
                            trial_params[param_name] = trial.suggest_categorical(
                                param_name, param_spec["values"])
                        elif param_spec.get("type") == "float":
                            trial_params[param_name] = trial.suggest_float(
                                param_name, param_spec["low"], param_spec["high"], 
                                log=param_spec.get("log", False))
                        elif param_spec.get("type") == "int":
                            trial_params[param_name] = trial.suggest_int(
                                param_name, param_spec["low"], param_spec["high"], 
                                log=param_spec.get("log", False))
                
                # Create a copy of the base config and update with trial parameters
                config_copy = copy.deepcopy(base_config)
                for param_name, param_value in trial_params.items():
                    if hasattr(config_copy, param_name):
                        setattr(config_copy, param_name, param_value)
                
                # Create environment
                env = environment_factory()
                
                # Evaluate with the provided evaluation function
                score = eval_func(config_copy, env)
                
                # Log trial results
                logger.info(f"Trial {trial.number}: score={score:.5f}, params={trial_params}")
                
                return score
            
            # Run the optimization
            self.optuna_optimizer.study.optimize(objective, n_trials=n_trials)
            
            # Store best parameters
            self.best_params = self.optuna_optimizer.study.best_params
            self.optimization_history = [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                    "datetime": trial.datetime_start.isoformat()
                }
                for trial in self.optuna_optimizer.study.trials
            ]
        
        logger.info(f"Pathfinder optimization completed. Best params: {self.best_params}")
        return self.best_params
    
    def distributed_train_q_learning(self,
                                    env_creator: Callable,
                                    base_config: Any,
                                    param_variations: List[Dict[str, Any]],
                                    episodes: int = 100,
                                    eval_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        Distribute Q-learning training across multiple workers using Ray.
        
        Args:
            env_creator: Function that creates environments.
            base_config: Base configuration for agents.
            param_variations: List of parameter variations to try.
            episodes: Number of episodes for training.
            eval_episodes: Number of episodes for evaluation.
            
        Returns:
            List[Dict]: Results from all distributed training runs.
        """
        if not self.ray_distributor.is_initialized:
            logger.warning("Ray not initialized. Running in sequential mode.")
        
        results = self.ray_distributor.distribute_q_learning(
            env_creator=env_creator,
            base_config=base_config,
            param_variations=param_variations,
            episodes=episodes,
            eval_episodes=eval_episodes
        )
        
        return results
    
    def distributed_train_maml(self,
                              task_creators: List[Callable],
                              test_task_creators: List[Callable],
                              base_maml_config: Any,
                              base_system_config: Any,
                              param_variations: List[Dict[str, Any]],
                              num_meta_iterations: int = 10,
                              num_adaptation_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Distribute MAML training across multiple workers using Ray.
        
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
        if not self.ray_distributor.is_initialized:
            logger.warning("Ray not initialized. Running in sequential mode.")
        
        results = self.ray_distributor.distribute_maml(
            task_creators=task_creators,
            test_task_creators=test_task_creators,
            base_maml_config=base_maml_config,
            base_system_config=base_system_config,
            param_variations=param_variations,
            num_meta_iterations=num_meta_iterations,
            num_adaptation_steps=num_adaptation_steps
        )
        
        return results
    
    def population_based_training(self,
                                 model_factory: Callable,
                                 env_creator: Callable,
                                 param_space: Dict[str, Any],
                                 train_func: Callable,
                                 eval_func: Callable,
                                 population_size: int = 4,
                                 num_iterations: int = 10) -> Dict[str, Any]:
        """
        Use Population Based Training (PBT) for optimization with Ray.
        
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
        if not self.ray_distributor.is_initialized:
            logger.warning("Ray not initialized. Cannot use PBT.")
            return {}
        
        best_params = self.ray_distributor.pbt_optimize(
            model_factory=model_factory,
            env_creator=env_creator,
            param_space=param_space,
            train_func=train_func,
            eval_func=eval_func,
            population_size=population_size,
            num_iterations=num_iterations
        )
        
        self.best_params = best_params
        return best_params
    
    def save_results(self, file_path: str):
        """
        Save optimization results to a file.
        
        Args:
            file_path (str): Path to save the results to.
        """
        # Check if we have results to save
        if self.best_params is None:
            logger.warning("No optimization results to save")
            return
                
        # Prepare results dictionary
        results = {
            "best_params": self.best_params,
            "best_value": getattr(self.optuna_optimizer.study, "best_value", None),
            "optimization_history": self.optimization_history,
            "datetime": datetime.datetime.now().isoformat(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else vars(self.config)
        }
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory:  # Only try to create directory if there is one
            os.makedirs(directory, exist_ok=True)
        
        # Save to file
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
            
        logger.info(f"Optimization results saved to {file_path}")
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load optimization results from a file.
        
        Args:
            file_path (str): Path to load the results from.
            
        Returns:
            Dict: The loaded optimization results.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Results file {file_path} not found")
            return None
            
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                
            self.best_params = results.get("best_params")
            self.optimization_history = results.get("optimization_history", [])
            
            logger.info(f"Loaded optimization results from {file_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None
    
    def shutdown(self):
        """Shut down Ray if it is initialized."""
        self.ray_distributor.shutdown()
        logger.info("Optimizer shutdown completed")
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Get the importance of each hyperparameter from the optimization history.
        
        Returns:
            Dict[str, float]: Dictionary mapping parameter names to importance scores.
        """
        if not hasattr(self.optuna_optimizer, "study") or self.optuna_optimizer.study is None:
            logger.warning("No study available for parameter importance analysis")
            return {}
            
        return self.optuna_optimizer.get_parameter_importance()
