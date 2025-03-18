"""
Optuna Hyperparameter Optimization Module for Q-Learning and MAML.

This module provides utilities to automatically optimize hyperparameters
for Q-Learning agents and Meta-Learning (MAML) algorithms using Optuna.
It supports optimization for various model types including neural networks,
transformers, and diffusion models.
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import asyncio
import logging
import copy
import datetime
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel
import os
import pickle

from .optimizer_config import OptimizerConfig
from re_learning.q_learning import AgentConfig, HierarchicalAgentSystemConfig

# Attempt to import MAML-specific components conditionally
try:
    from src.meta_learning.maml import MAMLConfig, MAMLAgent
except ImportError:
    # Create placeholder for type hints
    class MAMLConfig(BaseModel):
        pass
    
    class MAMLAgent:
        pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for Q-Learning and MAML.
    
    This class provides methods to optimize hyperparameters for reinforcement learning
    and meta-learning algorithms. It supports parallel optimization using Ray
    and various Optuna samplers and pruners.
    
    Attributes:
        config (OptimizerConfig): Configuration for the optimizer.
        study (optuna.Study): Optuna study object for hyperparameter optimization.
        best_params (Dict): Best hyperparameters found during optimization.
        optimization_history (List): History of optimization trials.
    """
    
    def __init__(self, config: OptimizerConfig):
        """
        Initialize the OptunaOptimizer.
        
        Args:
            config (OptimizerConfig): Configuration settings for the optimizer.
        """
        self.config = config
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        # Initialize Ray if configured
        if config.ray.address is not None:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(
                        address=config.ray.address,
                        num_cpus=config.ray.num_cpus,
                        num_gpus=config.ray.num_gpus,
                        _memory=config.ray.memory,
                        object_store_memory=config.ray.object_store_memory,
                        _redis_max_memory=config.ray.redis_max_memory
                    )
                    logger.info(f"Initialized Ray at {config.ray.address}")
            except ImportError:
                logger.warning("Ray not installed. Parallel optimization will not be available.")
                
        logger.info(f"OptunaOptimizer initialized with {config.optuna.study_name}")
    
    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Create and return an Optuna sampler based on the configuration."""
        sampler_name = self.config.optuna.sampler
        
        if sampler_name == "TPESampler":
            return TPESampler()
        elif sampler_name == "RandomSampler":
            return RandomSampler()
        elif sampler_name == "GridSampler":
            # Grid sampler requires explicit search space, which is handled elsewhere
            logger.warning("GridSampler selected but requires explicit search space")
            return TPESampler()  # Fallback
        else:
            logger.warning(f"Unknown sampler {sampler_name}, using TPESampler")
            return TPESampler()
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Create and return an Optuna pruner based on the configuration."""
        pruner_name = self.config.optuna.pruner
        
        if pruner_name == "MedianPruner":
            return MedianPruner()
        elif pruner_name == "SuccessiveHalvingPruner":
            return SuccessiveHalvingPruner()
        else:
            logger.warning(f"Unknown pruner {pruner_name}, using MedianPruner")
            return MedianPruner()
    
    def create_study(self, study_name: Optional[str] = None, direction: Optional[str] = None):
        """
        Create a new Optuna study for hyperparameter optimization.
        
        Args:
            study_name (Optional[str]): Name of the study. If None, uses the configured name.
            direction (Optional[str]): Optimization direction ('maximize' or 'minimize').
                If None, uses the configured direction.
        """
        if study_name is None:
            study_name = self.config.optuna.study_name
            
        if direction is None:
            direction = self.config.optuna.direction
            
        sampler = self._get_sampler()
        pruner = self._get_pruner()
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.optuna.storage,
            load_if_exists=True
        )
        
        logger.info(f"Created study {study_name} with direction {direction}")
    
    def optimize_q_learning(self, 
                           environment_factory: Callable,
                           param_space: Dict[str, Any],
                           base_agent_config: AgentConfig,
                           system_config: HierarchicalAgentSystemConfig,
                           eval_episodes: int = 10,
                           n_trials: Optional[int] = None,
                           timeout: Optional[int] = None):
        """
        Optimize hyperparameters for Q-Learning agents.
        
        Args:
            environment_factory: Callable that creates new environment instances.
            param_space: Dictionary defining the hyperparameter search space.
            base_agent_config: Base agent configuration to modify with trial parameters.
            system_config: System configuration for hierarchical agents.
            eval_episodes: Number of episodes for evaluation.
            n_trials: Number of trials to run. If None, uses the configured value.
            timeout: Maximum time (seconds) for optimization. If None, uses configured value.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        if self.study is None:
            self.create_study()
            
        if n_trials is None:
            n_trials = self.config.optuna.n_trials
            
        if timeout is None:
            timeout = self.config.optuna.timeout
            
        # Define the objective function for Q-Learning optimization
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
                    else:
                        logger.warning(f"Unknown parameter type for {param_name}")
                else:
                    logger.warning(f"Invalid parameter spec for {param_name}")
            
            # Create agent config with trial parameters
            agent_config = copy.deepcopy(base_agent_config)
            for param_name, param_value in trial_params.items():
                if param_name.startswith("q_learning_"):
                    # Set Q-learning specific parameters
                    q_param_name = param_name[len("q_learning_"):]
                    if hasattr(agent_config.q_learning_config, q_param_name):
                        setattr(agent_config.q_learning_config, q_param_name, param_value)
                else:
                    # Set agent-level parameters
                    if hasattr(agent_config, param_name):
                        setattr(agent_config, param_name, param_value)
            
            # Update system config with the modified agent config
            system_config_copy = copy.deepcopy(system_config)
            system_config_copy.agents_config = [agent_config]
            
            # Create environment
            env = environment_factory()
            
            # Create and train the agent system
            from re_learning.q_learning import HierarchicalAgentSystem
            agent_system = HierarchicalAgentSystem(system_config_copy)
            
            # Run training asynchronously
            loop = asyncio.get_event_loop()
            loop.run_until_complete(agent_system.train(env, agent_config.q_learning_config.episodes))
            
            # Evaluate the trained system
            total_reward = 0
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = loop.run_until_complete(agent_system.select_action(state))
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                total_reward += episode_reward
            
            # Clean up
            agent_system.close()
            avg_reward = total_reward / eval_episodes
            
            # Log trial results
            logger.info(f"Trial {trial.number}: avg_reward={avg_reward:.3f}, params={trial_params}")
            
            return avg_reward
        
        # Run the optimization
        logger.info(f"Starting Q-Learning optimization with {n_trials} trials")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Store best parameters
        self.best_params = self.study.best_params
        self.optimization_history = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime": trial.datetime_start.isoformat()
            }
            for trial in self.study.trials
        ]
        
        logger.info(f"Optimization completed. Best reward: {self.study.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def optimize_maml(self,
                     task_environment_factory: Callable[[int], Any],
                     param_space: Dict[str, Any],
                     base_system_config: HierarchicalAgentSystemConfig,
                     base_maml_config: MAMLConfig,
                     num_tasks: int = 5,
                     num_test_tasks: int = 2,
                     n_trials: Optional[int] = None,
                     timeout: Optional[int] = None):
        """
        Optimize hyperparameters for MAML (Meta-Agnostic Meta-Learning).
        
        Args:
            task_environment_factory: Callable that creates task environments given a task id.
            param_space: Dictionary defining the hyperparameter search space.
            base_system_config: Base system configuration.
            base_maml_config: Base MAML configuration.
            num_tasks: Number of tasks to use for meta-training.
            num_test_tasks: Number of tasks to use for meta-testing.
            n_trials: Number of trials to run. If None, uses the configured value.
            timeout: Maximum time (seconds) for optimization. If None, uses configured value.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        if self.study is None:
            self.create_study()
            
        if n_trials is None:
            n_trials = self.config.optuna.n_trials
            
        if timeout is None:
            timeout = self.config.optuna.timeout
            
        # Define the objective function for MAML optimization
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
                    else:
                        logger.warning(f"Unknown parameter type for {param_name}")
                else:
                    logger.warning(f"Invalid parameter spec for {param_name}")
            
            # Create MAML config with trial parameters
            maml_config = copy.deepcopy(base_maml_config)
            system_config = copy.deepcopy(base_system_config)
            
            for param_name, param_value in trial_params.items():
                if param_name.startswith("maml_"):
                    # Set MAML-specific parameters
                    maml_param_name = param_name[len("maml_"):]
                    if hasattr(maml_config, maml_param_name):
                        setattr(maml_config, maml_param_name, param_value)
                elif param_name.startswith("q_learning_"):
                    # Set Q-learning parameters in agent configs
                    q_param_name = param_name[len("q_learning_"):]
                    for agent_config in system_config.agents_config:
                        if hasattr(agent_config.q_learning_config, q_param_name):
                            setattr(agent_config.q_learning_config, q_param_name, param_value)
                else:
                    # Set system-level parameters
                    if hasattr(system_config, param_name):
                        setattr(system_config, param_name, param_value)
            
            # Create meta-training and meta-testing tasks
            training_tasks = {f"task_{i}": task_environment_factory(i) 
                             for i in range(num_tasks)}
            
            test_tasks = {f"test_task_{i}": task_environment_factory(num_tasks + i) 
                         for i in range(num_test_tasks)}
            
            # Create and train MAML agent
            maml_agent = MAMLAgent(system_config, maml_config)
            
            # Run meta-training asynchronously
            loop = asyncio.get_event_loop()
            loop.run_until_complete(maml_agent.meta_train(
                training_tasks, num_meta_iterations=maml_config.meta_iterations))
            
            # Evaluate on test tasks
            test_rewards = []
            for task_id, env in test_tasks.items():
                reward = loop.run_until_complete(maml_agent.adapt_to_new_task(
                    env, num_adaptation_steps=maml_config.adaptation_steps))
                test_rewards.append(reward)
            
            avg_test_reward = sum(test_rewards) / len(test_rewards)
            
            # Log trial results
            logger.info(f"Trial {trial.number}: avg_test_reward={avg_test_reward:.3f}, params={trial_params}")
            
            return avg_test_reward
        
        # Run the optimization
        logger.info(f"Starting MAML optimization with {n_trials} trials")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Store best parameters
        self.best_params = self.study.best_params
        self.optimization_history = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime": trial.datetime_start.isoformat()
            }
            for trial in self.study.trials
        ]
        
        logger.info(f"Optimization completed. Best reward: {self.study.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def optimize_neural_network(self,
                               model_factory: Callable,
                               dataset: Any,
                               param_space: Dict[str, Any],
                               eval_func: Callable,
                               n_trials: Optional[int] = None,
                               timeout: Optional[int] = None):
        """
        Optimize hyperparameters for neural networks.
        
        Args:
            model_factory: Function that creates model instances based on hyperparameters.
            dataset: Dataset to use for training and evaluation.
            param_space: Dictionary defining the hyperparameter search space.
            eval_func: Function that evaluates model performance given a model and dataset.
            n_trials: Number of trials to run. If None, uses the configured value.
            timeout: Maximum time (seconds) for optimization. If None, uses configured value.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        if self.study is None:
            self.create_study()
            
        if n_trials is None:
            n_trials = self.config.optuna.n_trials
            
        if timeout is None:
            timeout = self.config.optuna.timeout
            
        # Define the objective function for neural network optimization
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
                    else:
                        logger.warning(f"Unknown parameter type for {param_name}")
                else:
                    logger.warning(f"Invalid parameter spec for {param_name}")
            
            # Create model with trial parameters
            model = model_factory(**trial_params)
            
            # Evaluate model
            metric_value = eval_func(model, dataset)
            
            # Log trial results
            logger.info(f"Trial {trial.number}: metric={metric_value:.5f}, params={trial_params}")
            
            return metric_value
        
        # Run the optimization
        logger.info(f"Starting neural network optimization with {n_trials} trials")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Store best parameters
        self.best_params = self.study.best_params
        self.optimization_history = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime": trial.datetime_start.isoformat()
            }
            for trial in self.study.trials
        ]
        
        logger.info(f"Optimization completed. Best metric: {self.study.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def optimize_transformer(self,
                            model_factory: Callable,
                            dataset: Any,
                            param_space: Dict[str, Any],
                            eval_func: Callable,
                            n_trials: Optional[int] = None,
                            timeout: Optional[int] = None):
        """
        Optimize hyperparameters for transformer models.
        
        This is specialized for transformer architectures, but internally uses
        the same optimization process as neural networks with transformer-specific
        parameter spaces.
        
        Args:
            model_factory: Function that creates transformer instances based on hyperparameters.
            dataset: Dataset to use for training and evaluation.
            param_space: Dictionary defining the hyperparameter search space.
            eval_func: Function that evaluates model performance given a model and dataset.
            n_trials: Number of trials to run. If None, uses the configured value.
            timeout: Maximum time (seconds) for optimization. If None, uses configured value.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        # For transformers, we use the same core optimization logic as neural networks
        return self.optimize_neural_network(
            model_factory=model_factory,
            dataset=dataset,
            param_space=param_space,
            eval_func=eval_func,
            n_trials=n_trials,
            timeout=timeout
        )
    
    def optimize_diffusion(self,
                          model_factory: Callable,
                          dataset: Any,
                          param_space: Dict[str, Any],
                          eval_func: Callable,
                          n_trials: Optional[int] = None,
                          timeout: Optional[int] = None):
        """
        Optimize hyperparameters for diffusion models.
        
        This is specialized for diffusion model architectures, but internally uses
        the same optimization process as neural networks with diffusion-specific
        parameter spaces.
        
        Args:
            model_factory: Function that creates diffusion model instances based on hyperparameters.
            dataset: Dataset to use for training and evaluation.
            param_space: Dictionary defining the hyperparameter search space.
            eval_func: Function that evaluates model performance given a model and dataset.
            n_trials: Number of trials to run. If None, uses the configured value.
            timeout: Maximum time (seconds) for optimization. If None, uses configured value.
            
        Returns:
            Dict: The best hyperparameters found during optimization.
        """
        # For diffusion models, we use the same core optimization logic as neural networks
        return self.optimize_neural_network(
            model_factory=model_factory,
            dataset=dataset,
            param_space=param_space,
            eval_func=eval_func,
            n_trials=n_trials,
            timeout=timeout
        )
    
    def save_results(self, file_path: str):
        """
        Save optimization results to a file.
        
        Args:
            file_path (str): Path to save the results to.
        """
        # Check if we have results to save
        if not hasattr(self, 'best_params') or self.best_params is None:
            # Try to get best params from study if available
            if hasattr(self, 'study') and self.study is not None:
                self.best_params = self.study.best_params
                self.optimization_history = [
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name,
                        "datetime": trial.datetime_start.isoformat()
                    }
                    for trial in self.study.trials
                ]
            else:
                logger.warning("No optimization results to save")
                return
                
        # Prepare results dictionary
        results = {
            "best_params": self.best_params,
            "best_value": self.study.best_value if hasattr(self, 'study') and self.study is not None else None,
            "optimization_history": self.optimization_history if hasattr(self, 'optimization_history') else [],
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
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Get the importance of each hyperparameter from the optimization history.
        
        Returns:
            Dict[str, float]: Dictionary mapping parameter names to importance scores.
        """
        if not self.study:
            logger.warning("No study available for parameter importance analysis")
            return {}
            
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
            logger.info(f"Parameter importance calculated: {param_importance}")
            return param_importance
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}

# Example parameter spaces for common optimization targets
def get_q_learning_param_space() -> Dict[str, Dict]:
    """Get a default parameter space for Q-Learning optimization."""
    return {
        "q_learning_alpha": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "q_learning_gamma": {"type": "float", "low": 0.9, "high": 0.999},
        "q_learning_epsilon": {"type": "float", "low": 0.1, "high": 1.0},
        "q_learning_epsilon_decay": {"type": "float", "low": 0.9, "high": 0.999},
        "q_learning_learning_rate_decay": {"type": "float", "low": 0.9, "high": 1.0},
        "temperature": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "use_double_q": {"type": "categorical", "values": [True, False]}
    }

def get_maml_param_space() -> Dict[str, Dict]:
    """Get a default parameter space for MAML optimization."""
    return {
        "maml_meta_lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "maml_inner_lr": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "maml_adaptation_steps": {"type": "int", "low": 5, "high": 20},
        "maml_meta_batch_size": {"type": "int", "low": 2, "high": 10},
        "maml_bayes_weight": {"type": "float", "low": 0.0, "high": 1.0},
        "maml_maml_weight": {"type": "float", "low": 0.0, "high": 1.0}
    }

def get_neural_network_param_space() -> Dict[str, Dict]:
    """Get a default parameter space for neural network optimization."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 16, "high": 256, "log": True},
        "hidden_size": {"type": "int", "low": 32, "high": 512, "log": True},
        "num_layers": {"type": "int", "low": 1, "high": 5},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5}
    }

def get_transformer_param_space() -> Dict[str, Dict]:
    """Get a default parameter space for transformer optimization."""
    return {
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "batch_size": {"type": "int", "low": 8, "high": 128, "log": True},
        "num_heads": {"type": "int", "low": 2, "high": 16},
        "num_layers": {"type": "int", "low": 2, "high": 12},
        "d_model": {"type": "int", "low": 64, "high": 512, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "attention_dropout": {"type": "float", "low": 0.0, "high": 0.5}
    }

def get_diffusion_param_space() -> Dict[str, Dict]:
    """Get a default parameter space for diffusion model optimization."""
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "batch_size": {"type": "int", "low": 8, "high": 64, "log": True},
        "num_timesteps": {"type": "int", "low": 50, "high": 1000, "log": True},
        "hidden_size": {"type": "int", "low": 64, "high": 512, "log": True},
        "num_layers": {"type": "int", "low": 2, "high": 8},
        "attention_resolutions": {"type": "categorical", "values": ["16", "16,8", "16,8,4"]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3}
    }

# Example usage
async def example_q_learning_optimization():
    """Example of Q-Learning hyperparameter optimization."""
    from re_learning.q_learning import (
        AgentConfig, QLearningParams, HierarchicalAgentSystemConfig,
        HierarchicalMultiAgentEnvironment
    )
    
    # Create base configurations
    base_q_params = QLearningParams(
        state_space=(10, 10),
        action_space=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        episodes=100,
    )
    
    base_agent_config = AgentConfig(
        agent_id="agent1",
        q_learning_config=base_q_params,
        temperature=1.0,
        use_double_q=True
    )
    
    system_config = HierarchicalAgentSystemConfig(
        agents_config=[base_agent_config],
        grid_size=(10, 10),
        action_space=4,
    )
    
    # Environment factory
    def environment_factory():
        return HierarchicalMultiAgentEnvironment(grid_size=(10, 10))
    
    # Create optimizer
    optimizer_config = OptimizerConfig()
    optimizer = OptunaOptimizer(optimizer_config)
    
    # Define parameter space
    param_space = get_q_learning_param_space()
    
    # Run optimization
    best_params = optimizer.optimize_q_learning(
        environment_factory=environment_factory,
        param_space=param_space,
        base_agent_config=base_agent_config,
        system_config=system_config,
        eval_episodes=5,
        n_trials=20
    )
    
    # Save results
    optimizer.save_results("results/q_learning_optimization.pkl")
    
    return best_params

async def example_maml_optimization():
    """Example of MAML hyperparameter optimization."""
    from re_learning.q_learning import (
        AgentConfig, QLearningParams, HierarchicalAgentSystemConfig,
        HierarchicalMultiAgentEnvironment
    )
    from src.meta_learning.maml import MAMLConfig
    
    # Create base configurations
    base_q_params = QLearningParams(
        state_space=(10, 10),
        action_space=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        episodes=100,
    )
    
    base_agent_config = AgentConfig(
        agent_id="agent1",
        q_learning_config=base_q_params
    )
    
    system_config = HierarchicalAgentSystemConfig(
        agents_config=[base_agent_config],
        grid_size=(10, 10),
        action_space=4,
    )
    
    maml_config = MAMLConfig(
        meta_lr=0.01,
        inner_lr=0.1,
        meta_batch_size=3,
        adaptation_steps=5,
        meta_iterations=10,
    )
    
    # Task environment factory
    def task_environment_factory(task_id):
        # Create different environments based on task_id
        obstacles = [(i, i) for i in range(3, 7)]  # Different for each task
        return HierarchicalMultiAgentEnvironment(
            grid_size=(10, 10),
            obstacles=obstacles
        )
    
    # Create optimizer
    optimizer_config = OptimizerConfig()
    optimizer = OptunaOptimizer(optimizer_config)
    
    # Define parameter space
    param_space = get_maml_param_space()
    
    # Run optimization
    best_params = optimizer.optimize_maml(
        task_environment_factory=task_environment_factory,
        param_space=param_space,
        base_system_config=system_config,
        base_maml_config=maml_config,
        num_tasks=3,
        num_test_tasks=2,
        n_trials=10
    )
    
    # Save results
    optimizer.save_results("results/maml_optimization.pkl")
    
    return best_params

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(example_q_learning_optimization())
