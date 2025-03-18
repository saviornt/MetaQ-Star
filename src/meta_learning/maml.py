"""
MetaQ-Star: MCB-MAML Implementation

This module implements a Combined MMAML and BAMAML approach (MCB-MAML) for meta-reinforcement
learning with Q-learning agents. It builds upon the existing Q-learning and pathfinder
modules to provide a meta-learning framework that can quickly adapt to new tasks.

The implementation uses PyTorch for tensor operations and Numba for CPU optimizations
where applicable.

## Usage

```python
# Import the necessary components
from src.maml import MAMLAgent, MAMLConfig
from re_learning.q_learning import AgentConfig, QLearningParams, HierarchicalAgentSystemConfig

# Create configuration for the agent system
system_config = HierarchicalAgentSystemConfig(
    agents_config=[
        AgentConfig(
            agent_id="agent1",
            q_learning_config=QLearningParams(
                state_space=(10, 10),
                action_space=4,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                episodes=100,
            )
        )
    ],
    grid_size=(10, 10),
    action_space=4,
)

# Configure the MCB-MAML meta-learning parameters
mcb_config = MAMLConfig(
    meta_lr=0.01,
    inner_lr=0.1,
    meta_batch_size=5,
    adaptation_steps=10,
    meta_iterations=100,
)

# Create the MAMLAgent
mcb_agent = MAMLAgent(system_config, mcb_config)

# Prepare task environments
task_environments = {
    "task1": environment1,
    "task2": environment2,
    # ...
}

# Train the meta-learning agent
await mcb_agent.meta_train(task_environments)

# Adapt to a new task
eval_reward = await mcb_agent.adapt_to_new_task(new_environment)
```

See the `main()` function for a more complete example.

"""

import numpy as np
import torch
import asyncio
import logging
import copy
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pydantic import BaseModel, Field
from collections import defaultdict
from dataclasses import dataclass, field

from re_learning.q_learning import (
    HierarchicalAgentSystemConfig,
    MAMLQAgentSystem, Experience,
    DEVICE
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"MCB-MAML module initialized with device: {DEVICE}")

class MAMLConfig(BaseModel):
    """
    Configuration for the MCB-MAML (Combined MMAML and BAMAML) approach.
    
    This configuration class defines the hyperparameters and settings for the 
    Meta-Curvature Bayesian Model-Agnostic Meta-Learning algorithm, which combines
    the strengths of both MAML and Bayesian approaches to meta-learning.
    
    Attributes:
        meta_lr (float): Learning rate for meta-update (outer loop). Controls how
            quickly the meta-parameters are updated across tasks.
        inner_lr (float): Learning rate for task adaptation (inner loop). Controls
            how quickly the agent adapts to specific tasks.
        meta_batch_size (int): Number of tasks to sample for each meta-update iteration.
        adaptation_steps (int): Number of gradient steps for task adaptation in the inner loop.
        meta_iterations (int): Total number of meta-iterations to perform during training.
        prior_variance (float): Variance for the prior distribution in Bayesian component.
        likelihood_variance (float): Variance for the likelihood distribution.
        use_diagonal_approximation (bool): Whether to use diagonal approximation for covariance
            matrices, which is more computationally efficient for large parameter spaces.
        bayes_weight (float): Weight for the Bayesian component in the combined approach.
        maml_weight (float): Weight for the MAML component in the combined approach.
        task_sampling_strategy (str): Strategy for sampling tasks ('random', 'curriculum', 'active').
        use_parallel_adaptation (bool): Whether to parallelize adaptation across tasks.
        max_threads (int): Maximum number of threads for parallel adaptation.
        eval_interval (int): Interval (in meta-iterations) for evaluation.
        eval_episodes (int): Number of episodes for evaluation.
    """
    # Meta-learning parameters
    meta_lr: float = Field(
        default=0.01,
        description="Learning rate for meta-update (outer loop)"
    )
    inner_lr: float = Field(
        default=0.1,
        description="Learning rate for task adaptation (inner loop)"
    )
    meta_batch_size: int = Field(
        default=5,
        description="Number of tasks to sample for meta-update"
    )
    adaptation_steps: int = Field(
        default=10,
        description="Number of gradient steps for task adaptation"
    )
    meta_iterations: int = Field(
        default=100,
        description="Number of meta-iterations"
    )
    
    # Bayesian parameters
    prior_variance: float = Field(
        default=1.0,
        description="Variance for prior distribution"
    )
    likelihood_variance: float = Field(
        default=0.1,
        description="Variance for likelihood distribution"
    )
    use_diagonal_approximation: bool = Field(
        default=True,
        description="Whether to use diagonal approximation for covariance matrices"
    )
    
    # MCB-MAML specific parameters
    bayes_weight: float = Field(
        default=0.5,
        description="Weight for Bayesian component in the combined approach"
    )
    maml_weight: float = Field(
        default=0.5,
        description="Weight for MAML component in the combined approach"
    )
    
    # Task sampling parameters
    task_sampling_strategy: str = Field(
        default="random",
        description="Strategy for sampling tasks: 'random', 'curriculum', 'active'"
    )
    
    # Performance optimization
    use_parallel_adaptation: bool = Field(
        default=True,
        description="Whether to parallelize adaptation across tasks"
    )
    max_threads: int = Field(
        default=4,
        description="Maximum number of threads for parallel adaptation"
    )
    
    # Evaluation
    eval_interval: int = Field(
        default=10,
        description="Interval (in meta-iterations) for evaluation"
    )
    eval_episodes: int = Field(
        default=10,
        description="Number of episodes for evaluation"
    )

class MAMLTask:
    """Represents a single task for meta-learning."""
    
    def __init__(self, 
                 environment, 
                 task_id: str, 
                 config: MAMLConfig):
        self.environment = environment
        self.task_id = task_id
        self.config = config
        self.episode_rewards = []
        self.adaptation_trajectories = []
        self.posterior_mean = None
        self.posterior_variance = None
        
    def add_episode_reward(self, reward: float):
        """Add a reward from a completed episode."""
        self.episode_rewards.append(reward)
        
    def add_adaptation_trajectory(self, trajectory: List[Experience]):
        """Add a trajectory from adaptation."""
        self.adaptation_trajectories.append(trajectory)
        
    def get_average_reward(self) -> float:
        """Get the average reward across episodes."""
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)
    
    def reset_metrics(self):
        """Reset the task metrics."""
        self.episode_rewards = []
        self.adaptation_trajectories = []


class BayesianModelUpdater:
    """Handles Bayesian updates for the MCB-MAML approach."""
    
    def __init__(self, config: MAMLConfig):
        self.config = config
        self.prior_variance = config.prior_variance
        self.likelihood_variance = config.likelihood_variance
        self.use_diagonal_approximation = config.use_diagonal_approximation
        
    def initialize_prior(self, parameter_shapes: Dict[str, torch.Size]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize prior distribution for parameters."""
        prior = {}
        for name, shape in parameter_shapes.items():
            # Initialize mean at 0 and diagonal covariance matrix with prior_variance
            mean = torch.zeros(shape, device=DEVICE)
            if self.use_diagonal_approximation:
                # Use diagonal covariance representation
                variance = torch.ones(shape, device=DEVICE) * self.prior_variance
            else:
                # Full covariance matrix (for small parameters only)
                flat_size = np.prod(shape)
                if flat_size > 1000:
                    logger.warning(f"Using diagonal approximation for parameter {name} with size {flat_size}")
                    variance = torch.ones(shape, device=DEVICE) * self.prior_variance
                else:
                    variance = torch.eye(flat_size, device=DEVICE) * self.prior_variance
                    variance = variance.reshape(*shape, *shape)
            
            prior[name] = (mean, variance)
        
        return prior
    
    def compute_posterior(self, 
                          prior: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                          observations: Dict[str, List[torch.Tensor]]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute posterior distribution given prior and observations."""
        posterior = {}
        
        for name, (prior_mean, prior_variance) in prior.items():
            if name not in observations or not observations[name]:
                # No observations for this parameter, posterior = prior
                posterior[name] = (prior_mean, prior_variance)
                continue
                
            # Combine observations
            obs_tensor = torch.stack(observations[name])
            
            if self.use_diagonal_approximation:
                # Diagonal approximation for Bayesian update
                # Posterior precision = prior precision + n * likelihood precision
                n = len(observations[name])
                likelihood_precision = 1.0 / self.likelihood_variance
                prior_precision = 1.0 / prior_variance
                posterior_precision = prior_precision + n * likelihood_precision
                posterior_variance = 1.0 / posterior_precision
                
                # Posterior mean
                weighted_obs = torch.mean(obs_tensor, dim=0) * (n * likelihood_precision)
                weighted_prior = prior_mean * prior_precision
                posterior_mean = (weighted_prior + weighted_obs) / posterior_precision
                
            else:
                # Full covariance update (for small parameters only)
                # Would implement full Bayesian update with matrix operations
                # Simplified here to save space
                logger.warning("Full covariance update not fully implemented - using diagonal approximation")
                # Fall back to diagonal approximation
                n = len(observations[name])
                likelihood_precision = 1.0 / self.likelihood_variance
                prior_precision = 1.0 / prior_variance
                posterior_precision = prior_precision + n * likelihood_precision
                posterior_variance = 1.0 / posterior_precision
                
                # Posterior mean
                weighted_obs = torch.mean(obs_tensor, dim=0) * (n * likelihood_precision)
                weighted_prior = prior_mean * prior_precision
                posterior_mean = (weighted_prior + weighted_obs) / posterior_precision
            
            posterior[name] = (posterior_mean, posterior_variance)
            
        return posterior
    
    def sample_parameters(self, 
                          posterior: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Sample parameters from posterior distribution."""
        sampled_params = {}
        
        for name, (mean, variance) in posterior.items():
            if self.use_diagonal_approximation:
                # Sample from diagonal normal distribution
                std_dev = torch.sqrt(variance)
                noise = torch.randn_like(mean, device=DEVICE)
                sampled_params[name] = mean + noise * std_dev
            else:
                # Would implement full covariance sampling
                # Simplified for now
                std_dev = torch.sqrt(variance)
                noise = torch.randn_like(mean, device=DEVICE)
                sampled_params[name] = mean + noise * std_dev
                
        return sampled_params


class MAMLAgent(MAMLQAgentSystem):
    """
    Combined MMAML and BAMAML agent implementation (MCB-MAML).
    
    This agent system extends the MAMLQAgentSystem with Bayesian adaptation capabilities,
    allowing for uncertainty-aware meta-learning. It combines the strengths of MAML
    (fast adaptation through gradient-based meta-learning) with Bayesian approaches
    (uncertainty modeling and prior knowledge incorporation).
    
    The agent maintains a prior and posterior distribution over parameters and uses
    both MAML-style updates and Bayesian inference to adapt to new tasks.
    
    Args:
        system_config (HierarchicalAgentSystemConfig): Configuration for the hierarchical
            agent system, including agent configurations and environment parameters.
        mcb_config (MAMLConfig): Configuration specific to the MCB-MAML approach.
            
    Attributes:
        mcb_config (MAMLConfig): Configuration for the MCB-MAML approach.
        bayesian_updater (BayesianModelUpdater): Helper for Bayesian updates.
        prior (Dict): Prior distributions for all parameters.
        posterior (Dict): Posterior distributions for all parameters.
        tasks (Dict): Dictionary of registered tasks.
        meta_iteration (int): Current meta-training iteration.
        meta_training_rewards (List[float]): History of rewards during meta-training.
        adaptation_times (List[float]): History of adaptation times.
    """
    
    def __init__(self, 
                 system_config: HierarchicalAgentSystemConfig,
                 mcb_config: MAMLConfig):
        """Initialize the MCB-MAML agent system."""
        super().__init__(system_config)
        
        # MCB-MAML specific configuration
        self.mcb_config = mcb_config
        
        # Initialize Bayesian components
        self.bayesian_updater = BayesianModelUpdater(mcb_config)
        
        # Get parameter shapes for initialization
        parameter_shapes = self._get_parameter_shapes()
        
        # Initialize prior distributions for all parameters
        self.prior = self.bayesian_updater.initialize_prior(parameter_shapes)
        
        # Initialize posterior to be same as prior initially
        self.posterior = copy.deepcopy(self.prior)
        
        # Task-specific adaptation parameters
        self.task_specific_parameters = {}
        
        # Meta-learning state
        self.meta_iteration = 0
        self.tasks = {}  # Dict[task_id, MAMLTask]
        
        # Performance tracking
        self.meta_training_rewards = []
        self.adaptation_times = []
        
        logger.info(f"MCB-MAML agent initialized with {len(self.agents)} agents")
        
    def _get_parameter_shapes(self) -> Dict[str, torch.Size]:
        """Get shapes of all learnable parameters in the system."""
        shapes = {}
        
        # Add Q-value table shapes for each agent
        for agent_id, agent in self.agents.items():
            # Get shapes from q_table
            if hasattr(agent, 'q_table') and isinstance(agent.q_table, torch.Tensor):
                shapes[f"{agent_id}_q_table"] = agent.q_table.shape
                
            # If using DQN, add network parameters
            if hasattr(agent, 'policy_net'):
                for name, param in agent.policy_net.named_parameters():
                    shapes[f"{agent_id}_{name}"] = param.shape
                    
        return shapes
    
    def _extract_current_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract current parameters from all agents."""
        parameters = {}
        
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'q_table') and isinstance(agent.q_table, torch.Tensor):
                parameters[f"{agent_id}_q_table"] = agent.q_table.clone()
                
            if hasattr(agent, 'policy_net'):
                for name, param in agent.policy_net.named_parameters():
                    parameters[f"{agent_id}_{name}"] = param.clone().detach()
                    
        return parameters
    
    def _apply_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Apply parameters to all agents."""
        for agent_id, agent in self.agents.items():
            q_table_key = f"{agent_id}_q_table"
            if q_table_key in parameters and hasattr(agent, 'q_table'):
                agent.q_table = parameters[q_table_key].clone()
                
            if hasattr(agent, 'policy_net'):
                for name, param in agent.policy_net.named_parameters():
                    param_key = f"{agent_id}_{name}"
                    if param_key in parameters:
                        param.data = parameters[param_key].clone()

    async def register_task(self, environment, task_id: str) -> str:
        """Register a new task for meta-learning."""
        task = MAMLTask(environment, task_id, self.mcb_config)
        self.tasks[task_id] = task
        logger.info(f"Registered task {task_id}, total tasks: {len(self.tasks)}")
        return task_id
    
    async def meta_train(self, task_environments: Dict[str, Any], num_meta_iterations: Optional[int] = None):
        """
        Perform meta-training on a set of tasks.
        
        This method implements the main meta-learning loop for MCB-MAML, which involves:
        1. Sampling a batch of tasks
        2. Adapting parameters to each task (inner loop)
        3. Updating meta-parameters using both MAML and Bayesian approaches (outer loop)
        4. Periodically evaluating the meta-policy
        
        Args:
            task_environments (Dict[str, Any]): Dictionary mapping task_id to environment.
                Each environment should implement reset() and step() methods.
            num_meta_iterations (Optional[int]): Number of meta-iterations to perform.
                If None, uses the value from mcb_config.
                
        Returns:
            None
            
        Note:
            This method updates the agent's meta-parameters in-place and records
            performance metrics in meta_training_rewards and adaptation_times.
        """
        if num_meta_iterations is None:
            num_meta_iterations = self.mcb_config.meta_iterations
            
        # Register any unregistered tasks
        for task_id, env in task_environments.items():
            if task_id not in self.tasks:
                await self.register_task(env, task_id)
                
        logger.info(f"Starting meta-training for {num_meta_iterations} iterations " 
                   f"with {len(task_environments)} tasks")
        
        for iteration in range(num_meta_iterations):
            self.meta_iteration = iteration
            
            # Sample tasks according to strategy
            sampled_task_ids = self._sample_tasks(list(task_environments.keys()), 
                                                 self.mcb_config.meta_batch_size)
            
            # Store original meta-parameters
            original_params = self._extract_current_parameters()
            
            # Container for task-specific adapted parameters
            task_adaptations = {}
            task_gradients = []
            
            # Performance metrics for this iteration
            iteration_start_time = time.time()
            iteration_rewards = []
            
            # Adapt to each task (using parallel adaptation if enabled)
            if self.mcb_config.use_parallel_adaptation:
                # Parallel adaptation
                tasks = [self.tasks[task_id] for task_id in sampled_task_ids]
                envs = [task_environments[task_id] for task_id in sampled_task_ids]
                
                # Create adaptation coroutines
                adaptation_coros = [
                    self._adapt_to_task(task, env) 
                    for task, env in zip(tasks, envs)
                ]
                
                # Run adaptations in parallel
                adaptation_results = await asyncio.gather(*adaptation_coros)
                
                # Process results
                for task_id, adapted_params, task_grad, reward in adaptation_results:
                    task_adaptations[task_id] = adapted_params
                    task_gradients.append(task_grad)
                    iteration_rewards.append(reward)
            else:
                # Sequential adaptation
                for task_id in sampled_task_ids:
                    task = self.tasks[task_id]
                    env = task_environments[task_id]
                    
                    # Adapt to this specific task
                    adapted_params, task_grad, reward = await self._adapt_to_task(task, env)
                    
                    # Store adaptations
                    task_adaptations[task_id] = adapted_params
                    task_gradients.append(task_grad)
                    iteration_rewards.append(reward)
            
            # Update posterior distribution with new task observations
            observations = defaultdict(list)
            for task_params in task_adaptations.values():
                for name, param in task_params.items():
                    observations[name].append(param)
            
            self.posterior = self.bayesian_updater.compute_posterior(
                self.prior, 
                observations
            )
            
            # Compute combined update using both MAML and Bayesian components
            self._apply_meta_update(original_params, task_gradients)
            
            # Restore meta-parameters for next iteration
            self._apply_parameters(original_params)
            
            # Record metrics
            iteration_time = time.time() - iteration_start_time
            avg_reward = sum(iteration_rewards) / len(iteration_rewards)
            self.meta_training_rewards.append(avg_reward)
            self.adaptation_times.append(iteration_time)
            
            logger.info(f"Meta-iteration {iteration+1}/{num_meta_iterations}: "
                       f"avg_reward={avg_reward:.3f}, time={iteration_time:.2f}s")
            
            # Evaluation
            if (iteration + 1) % self.mcb_config.eval_interval == 0:
                await self._evaluate_meta_policy(task_environments)
    
    async def _adapt_to_task(self, 
                            task: MAMLTask, 
                            environment) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        """
        Adapt to a specific task using the inner loop update.
        
        Returns:
            Tuple containing:
            - Adapted parameters
            - Task gradients (for MAML update)
            - Average reward after adaptation
        """
        # Store original parameters for restoring later
        original_params = self._extract_current_parameters()
        
        # Sample parameters from posterior for Bayesian initialization
        sampled_posterior_params = self.bayesian_updater.sample_parameters(self.posterior)
        
        # Apply sampled parameters for initialization
        meta_params = {}
        for name, param in original_params.items():
            if name in sampled_posterior_params:
                # Combine MAML initialization with Bayesian sampling using the specified weights
                meta_params[name] = (
                    self.mcb_config.maml_weight * param + 
                    self.mcb_config.bayes_weight * sampled_posterior_params[name]
                )
            else:
                meta_params[name] = param
        
        # Apply the combined initialization parameters
        self._apply_parameters(meta_params)
        
        # Initialize task-specific parameters from meta-parameters
        task_params = copy.deepcopy(meta_params)
        
        # Inner loop adaptation
        total_reward = 0
        num_episodes = self.mcb_config.adaptation_steps
        
        for step in range(num_episodes):
            # Run an episode
            episode_reward, experiences = await self._run_adaptation_episode(environment)
            total_reward += episode_reward
            
            # Store experiences for this task
            if experiences:
                task.add_adaptation_trajectory(experiences)
            
            # Update task parameters based on experiences
            task_params = await self._update_task_parameters(task_params, experiences)
            
            # Apply updated parameters
            self._apply_parameters(task_params)
            
        # Calculate average reward
        avg_reward = total_reward / num_episodes
        task.add_episode_reward(avg_reward)
        
        # Calculate gradient as difference between adapted and original params
        task_gradient = {}
        for name, original_value in original_params.items():
            if name in task_params:
                task_gradient[name] = task_params[name] - original_value
        
        # Reset to original meta-parameters
        self._apply_parameters(original_params)
        
        return task_params, task_gradient, avg_reward
    
    async def _run_adaptation_episode(self, environment) -> Tuple[float, List[Experience]]:
        """Run a single episode for task adaptation."""
        state = environment.reset()
        done = False
        total_reward = 0
        experiences = []
        
        max_steps = 100  # Default max steps
        if hasattr(self.config.agents_config[0], 'q_learning_config'):
            max_steps = self.config.agents_config[0].q_learning_config.max_steps
        
        step_count = 0
        while not done and step_count < max_steps:
            # Select action
            action = await self.select_action(state)
            
            # Take action
            next_state, reward, done, info = environment.step(action)
            
            # Create experience
            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # Store experience
            experiences.append(exp)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Share experience with all agents
            await self._share_experience(exp)
            
            # Process agent messages
            await self._process_all_messages()
        
        return total_reward, experiences
    
    async def _update_task_parameters(self, 
                                     task_params: Dict[str, torch.Tensor], 
                                     experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Update task-specific parameters using experiences from an episode."""
        # Skip if no experiences
        if not experiences:
            return task_params
        
        # Process experiences with all agents
        for agent_id, agent in self.agents.items():
            # Filter experiences relevant to this agent
            agent_experiences = [
                exp for exp in experiences
                if self._is_state_in_domain(exp.state, agent.subtask_domain)
            ]
            
            if agent_experiences:
                # Have agent process these experiences
                worker_id = hash(agent_id) % self.config.max_learning_threads
                await asyncio.to_thread(agent.process_experiences, agent_experiences, worker_id)
        
        # Extract updated parameters
        updated_params = self._extract_current_parameters()
        
        # Calculate adapted parameters using inner loop learning rate
        adapted_params = {}
        for name, original_param in task_params.items():
            if name in updated_params:
                # Apply inner loop update: θ' = θ - α∇L(θ)
                # In our case, this is: θ' = θ + α(θ_new - θ)
                update = updated_params[name] - original_param
                adapted_params[name] = original_param + self.mcb_config.inner_lr * update
            else:
                adapted_params[name] = original_param
        
        return adapted_params
    
    def _apply_meta_update(self, 
                          original_params: Dict[str, torch.Tensor],
                          task_gradients: List[Dict[str, torch.Tensor]]):
        """Apply meta-update using both MAML and Bayesian components."""
        # Combine all task gradients for MAML update
        avg_gradients = {}
        for name in original_params.keys():
            # Get all gradients for this parameter
            param_grads = [
                grad[name] for grad in task_gradients 
                if name in grad
            ]
            
            if param_grads:
                # Compute average gradient across tasks
                avg_gradients[name] = torch.stack(param_grads).mean(dim=0)
        
        # Compute updated meta-parameters using MAML update rule
        maml_params = {}
        for name, param in original_params.items():
            if name in avg_gradients:
                maml_params[name] = param + self.mcb_config.meta_lr * avg_gradients[name]
            else:
                maml_params[name] = param
        
        # Sample from posterior for Bayesian component
        bayes_params = self.bayesian_updater.sample_parameters(self.posterior)
        
        # Combine MAML and Bayesian updates
        combined_params = {}
        for name, maml_param in maml_params.items():
            if name in bayes_params:
                # Weighted combination
                combined_params[name] = (
                    self.mcb_config.maml_weight * maml_param +
                    self.mcb_config.bayes_weight * bayes_params[name]
                )
            else:
                combined_params[name] = maml_param
        
        # Apply the combined update
        self._apply_parameters(combined_params)
    
    def _sample_tasks(self, task_ids: List[str], batch_size: int) -> List[str]:
        """Sample tasks according to the configured strategy."""
        if len(task_ids) <= batch_size:
            return task_ids
        
        strategy = self.mcb_config.task_sampling_strategy
        
        if strategy == "random":
            # Random sampling
            return np.random.choice(task_ids, size=batch_size, replace=False).tolist()
            
        elif strategy == "curriculum":
            # Curriculum learning - sample tasks with lowest performance
            # This promotes learning on difficult tasks
            task_rewards = {
                task_id: self.tasks[task_id].get_average_reward()
                for task_id in task_ids
                if task_id in self.tasks
            }
            
            # Sort by reward (ascending)
            sorted_tasks = sorted(task_rewards.items(), key=lambda x: x[1])
            return [task_id for task_id, _ in sorted_tasks[:batch_size]]
            
        elif strategy == "active":
            # Active learning - prioritize tasks with highest uncertainty
            # This would require uncertainty estimates per task
            # For now, we'll use a simple heuristic based on reward variance
            task_reward_vars = {}
            for task_id in task_ids:
                if task_id in self.tasks and len(self.tasks[task_id].episode_rewards) > 1:
                    rewards = self.tasks[task_id].episode_rewards
                    task_reward_vars[task_id] = np.var(rewards)
                else:
                    # If no data, assign high variance to encourage exploration
                    task_reward_vars[task_id] = float('inf')
            
            # Sort by variance (descending)
            sorted_tasks = sorted(task_reward_vars.items(), key=lambda x: -x[1])
            return [task_id for task_id, _ in sorted_tasks[:batch_size]]
            
        else:
            # Default to random
            return np.random.choice(task_ids, size=batch_size, replace=False).tolist()
    
    async def _evaluate_meta_policy(self, task_environments: Dict[str, Any]):
        """Evaluate current meta-policy on all tasks."""
        logger.info(f"Evaluating meta-policy at iteration {self.meta_iteration}")
        
        original_params = self._extract_current_parameters()
        evaluation_rewards = []
        
        for task_id, env in task_environments.items():
            # Evaluate without adaptation
            avg_reward = await self._evaluate_on_task(env, num_episodes=self.mcb_config.eval_episodes)
            evaluation_rewards.append(avg_reward)
            logger.info(f"Task {task_id} evaluation: reward={avg_reward:.3f}")
            
        avg_eval_reward = sum(evaluation_rewards) / len(evaluation_rewards)
        logger.info(f"Meta-evaluation: average reward={avg_eval_reward:.3f}")
        
        # Reset to original parameters
        self._apply_parameters(original_params)
        
        return avg_eval_reward
    
    async def _evaluate_on_task(self, environment, num_episodes: int = 10) -> float:
        """Evaluate current policy on a specific task."""
        total_reward = 0
        
        for episode in range(num_episodes):
            episode_reward, _ = await self._run_adaptation_episode(environment)
            total_reward += episode_reward
            
        return total_reward / num_episodes
    
    async def adapt_to_new_task(self, environment, num_adaptation_steps: Optional[int] = None) -> float:
        """
        Adapt to a new (unseen) task using the meta-learned policy.
        
        This method applies the learned meta-policy to adapt to a new task not seen
        during meta-training. It combines both MAML and Bayesian approaches by:
        1. Initializing parameters using a weighted combination of meta-parameters
           and samples from the Bayesian posterior
        2. Adapting these parameters to the new task through interaction with the environment
        3. Evaluating the adapted policy
        
        Args:
            environment: The environment for the new task. Should implement reset() and step().
            num_adaptation_steps (Optional[int]): Number of adaptation steps to perform.
                If None, uses the value from mcb_config.
                
        Returns:
            float: Average reward after adaptation, indicating performance on the new task.
            
        Note:
            This method modifies the agent's parameters in-place. To restore the
            original meta-parameters, they should be saved before calling this method.
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.mcb_config.adaptation_steps
            
        # Store original meta-parameters
        original_params = self._extract_current_parameters()
        
        # Sample parameters from posterior for Bayesian initialization
        sampled_posterior_params = self.bayesian_updater.sample_parameters(self.posterior)
        
        # Apply sampled parameters for initialization
        meta_params = {}
        for name, param in original_params.items():
            if name in sampled_posterior_params:
                # Combine MAML initialization with Bayesian sampling
                meta_params[name] = (
                    self.mcb_config.maml_weight * param + 
                    self.mcb_config.bayes_weight * sampled_posterior_params[name]
                )
            else:
                meta_params[name] = param
        
        # Apply the combined initialization parameters
        self._apply_parameters(meta_params)
        
        # Inner loop adaptation
        total_reward = 0
        all_experiences = []
        
        for step in range(num_adaptation_steps):
            # Run an episode
            episode_reward, experiences = await self._run_adaptation_episode(environment)
            total_reward += episode_reward
            all_experiences.extend(experiences)
            
            # Continually update using experiences
            current_params = self._extract_current_parameters()
            updated_params = await self._update_task_parameters(current_params, experiences)
            self._apply_parameters(updated_params)
        
        # Evaluate after adaptation
        eval_reward = await self._evaluate_on_task(environment, num_episodes=5)
        
        logger.info(f"Adaptation to new task completed: "
                   f"adaptation_reward={total_reward/num_adaptation_steps:.3f}, "
                   f"eval_reward={eval_reward:.3f}")
        
        return eval_reward
    
    def _is_state_in_domain(self, state: Tuple[int, int], domain: List[Tuple[int, int]]) -> bool:
        """Check if a state is within an agent's domain."""
        return state in domain

async def main():
    """
    Example usage of the MCB-MAML implementation.
    
    This function demonstrates how to:
    1. Configure a simple hierarchical agent system
    2. Set up the MCB-MAML agent
    3. Create task environments
    4. Perform meta-training
    5. Adapt to new tasks
    
    For a complete working example, additional environment setup would be needed.
    """
    from re_learning.q_learning import (
        AgentConfig, QLearningParams, HierarchicalAgentSystemConfig
    )
    
    # Example system configuration
    system_config = HierarchicalAgentSystemConfig(
        agents_config=[
            AgentConfig(
                agent_id="agent1",
                q_learning_config=QLearningParams(
                    state_space=(10, 10),
                    action_space=4,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=1.0,
                    episodes=100,
                )
            )
        ],
        grid_size=(10, 10),
        action_space=4,
    )
    
    # MCB-MAML configuration
    mcb_config = MAMLConfig(
        meta_lr=0.01,
        inner_lr=0.1,
        meta_batch_size=5,
        adaptation_steps=10,
        meta_iterations=100,
    )
    
    # Create MCB-MAML agent
    mcb_agent = MAMLAgent(system_config, mcb_config)
    
    # Task environments would be created here
    # task_environments = {...}
    
    # Then meta-training would be performed
    # await mcb_agent.meta_train(task_environments)
    
    # And adaptation to new task
    # await mcb_agent.adapt_to_new_task(new_environment)
    
    logger.info("MCB-MAML example completed")

if __name__ == "__main__":
    asyncio.run(main())
