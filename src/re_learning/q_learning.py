# ./src/re_learning/q_learning.py

import numpy as np
import asyncio
import threading
import random
import logging
import concurrent.futures
import os
import torch
from enum import Enum
from typing import Optional, List, Any, Tuple, Dict, Set, Union
from dataclasses import dataclass, field
from queue import Queue
from collections import deque
from numba import njit, prange
from pydantic import BaseModel, Field, model_validator
from pathfinder.pathfinder import AStarPathfinder, PathfinderConfig
from cache_manager.cache_manager import CacheManager, CacheConfig
from src.utils.resource_manager import ResourceManager, ResourceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up device selection based on availability
def get_device():
    """Get the best available device based on preference: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")

# Flag to determine if we should use CPU optimizations
USE_NUMBA = DEVICE.type == 'cpu'
logger.info(f"Using Numba optimizations: {USE_NUMBA}")

class CacheStorage(str, Enum):
    """
    Enumeration for available storage backends for the Q-table.
    """
    REDIS = "redis"
    MEMORY = "memory"

class PersistentStorage(str, Enum):
    """
    Enumeration for available persistent storage backends for the Q-table.
    """
    SQLITE = "sqlite"
    MONGO = "mongo"

class QLearningConfig(BaseModel):
    """
    Base configuration class for the Q-Learning agent.
    Encapsulates all configurations related to the Q-Learning algorithm.
    """
    alpha: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate for Q-Learning updates."
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for future rewards."
    )
    epsilon: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Initial exploration rate for epsilon-greedy strategy."
    )
    epsilon_min: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum exploration rate."
    )
    epsilon_decay: float = Field(
        default=0.995,
        ge=0.0,
        le=1.0,
        description="Decay rate for exploration probability."
    )
    episodes: int = Field(
        default=1000,
        ge=1,
        description="Total number of training episodes."
    )
    max_steps: int = Field(
        default=100,
        ge=1,
        description="Maximum steps per episode."
    )
    learning_rate_decay: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Decay rate for the learning rate alpha."
    )
    replay_buffer_size: int = Field(
        default=10000,
        ge=1,
        description="Maximum size of the experience replay buffer."
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Number of experiences sampled from the replay buffer for training."
    )
    heuristic_update_frequency: int = Field(
        default=50,
        ge=1,
        description="Frequency (in episodes) to perform heuristic-based updates."
    )
    heuristic_update_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight factor for heuristic-based Q-table updates."
    )
    exploration_strategy: str = Field(
        default="epsilon_greedy",
        description="Exploration strategy to use (e.g., 'epsilon_greedy', 'softmax')."
    )
    cache_storage: CacheStorage = Field(
        default=CacheStorage.REDIS,
        description="Cache storage for the Q-table ('redis' or 'memory')."
    )
    persistent_storage: PersistentStorage = Field(
        default=PersistentStorage.MONGO,
        description="Persistent storage for the Q-table ('sqlite' or 'mongo')."
    )
    # Additional configuration parameters can be added here as needed.

class QLearningParams(QLearningConfig):
    """
    Extended configuration class for the Q-Learning agent, including MAML-specific parameters.

    This class extends QLearningConfig with parameters specific to meta-learning and task adaptation.

    Args:
        state_space (Tuple[int, int]): Dimensions of the state space (e.g., grid size), default (10, 10).
        action_space (int): Number of possible actions, default 4.
        meta_lr (float): Learning rate for meta-update (outer loop in MAML), default 0.01.
        adapt_lr (float): Learning rate for task adaptation (inner loop in MAML), default 0.1.
        meta_batch_size (int): Number of tasks to sample for meta-update, default 5.
        adaptation_steps (int): Number of gradient steps for task adaptation, default 10.
        adaptation_episodes (int): Number of episodes for task adaptation, default 50.

    Additionally inherits all parameters from QLearningConfig (alpha, gamma, epsilon, etc.).
    """
    state_space: Tuple[int, int] = Field(
        default=(10, 10),
        description="Dimensions of the state space (e.g., grid size)."
    )
    action_space: int = Field(
        default=4,
        description="Number of possible actions."
    )
    # MAML-specific parameters
    meta_lr: float = Field(
        default=0.01,
        description="Learning rate for meta-update (outer loop in MAML)"
    )
    adapt_lr: float = Field(
        default=0.1,
        description="Learning rate for task adaptation (inner loop in MAML)"
    )
    meta_batch_size: int = Field(
        default=5,
        description="Number of tasks to sample for meta-update"
    )
    adaptation_steps: int = Field(
        default=10,
        description="Number of gradient steps for task adaptation"
    )
    adaptation_episodes: int = Field(
        default=50,
        description="Number of episodes for task adaptation"
    )
    # Additional initial hyperparameters for Optuna can be added here as needed.

class Experience(BaseModel):
    """Experience tuple for replay buffer."""
    state: Tuple[int, int]
    action: int
    reward: float
    next_state: Tuple[int, int]
    done: bool
    
    class Config:
        arbitrary_types_allowed = True

class AgentConfig(BaseModel):
    """
    Configuration for an individual agent within the hierarchical system.

    Args:
        agent_id (str): Unique identifier for the agent.
        q_learning_config (QLearningParams): Q-learning hyperparameters.
        subtask_domain (List[Tuple[int, int]]): State space subset this agent is responsible for, default [].
        communication_frequency (int): Frequency of communication with other agents (in episodes), default 5.
        temperature (float): Temperature parameter for softmax action selection, default 1.0.
        use_double_q (bool): Whether to use double Q-learning, default True.
        target_update_frequency (int): Frequency of target network updates (in episodes), default 100.
        heuristic_weight (float): Weight for heuristic-guided exploration, default 0.5.
    """
    agent_id: str
    q_learning_config: QLearningParams
    subtask_domain: List[Tuple[int, int]] = Field(
        default=[],
        description="State space subset this agent is responsible for"
    )
    communication_frequency: int = Field(
        default=5,
        description="Frequency of communication with other agents (in episodes)"
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature parameter for softmax action selection"
    )
    use_double_q: bool = Field(
        default=True,
        description="Whether to use double Q-learning"
    )
    target_update_frequency: int = Field(
        default=100,
        description="Frequency of target network updates (in episodes)"
    )
    heuristic_weight: float = Field(
        default=0.5,
        description="Weight for heuristic-guided exploration"
    )

class HierarchicalAgentSystemConfig(BaseModel):
    """
    Configuration for the entire hierarchical agent system.

    Args:
        agents_config (List[AgentConfig]): List of agent configurations.
        grid_size (Tuple[int, int]): Size of the grid world, default (100, 100).
        action_space (int): Number of possible actions (4 or 8 for grid world), default 8.
        pathfinder_config (Optional[PathfinderConfig]): Configuration for the pathfinder, default None.
        max_communication_threads (int): Maximum number of threads for agent communication, default 4.
        max_learning_threads (int): Maximum number of threads for parallel learning, default 4.
        experience_sharing_threshold (float): Minimum reward threshold for sharing experiences, default 0.5.
        coordination_frequency (int): Frequency of coordination between agents (in episodes), default 20.

    The class automatically validates agent configurations to ensure consistency with the system settings.
    """
    agents_config: List[AgentConfig]
    grid_size: Tuple[int, int] = Field(
        default=(100, 100),
        description="Size of the grid world"
    )
    action_space: int = Field(
        default=8,
        description="Number of possible actions (4 or 8 for grid world)"
    )
    pathfinder_config: Optional[PathfinderConfig] = None
    max_communication_threads: int = Field(
        default=4,
        description="Maximum number of threads for agent communication"
    )
    max_learning_threads: int = Field(
        default=4,
        description="Maximum number of threads for parallel learning"
    )
    experience_sharing_threshold: float = Field(
        default=0.5,
        description="Minimum reward threshold for sharing experiences"
    )
    coordination_frequency: int = Field(
        default=20,
        description="Frequency of coordination between agents (in episodes)"
    )
    
    @model_validator(mode='after')
    def validate_agent_configs(self):
        """Validate that agent configurations are consistent with the system configuration."""
        for agent_config in self.agents_config:
            if agent_config.q_learning_config.state_space != self.grid_size:
                agent_config.q_learning_config.state_space = self.grid_size
            if agent_config.q_learning_config.action_space != self.action_space:
                agent_config.q_learning_config.action_space = self.action_space
        return self

# PyTorch-based helper functions to replace Numba functions
def calculate_td_error(q_val: torch.Tensor, next_q_val: torch.Tensor, reward: torch.Tensor, gamma: float) -> torch.Tensor:
    """Calculate TD error using PyTorch."""
    return reward + gamma * next_q_val - q_val

def update_q_value(q_val: torch.Tensor, td_error: torch.Tensor, alpha: float) -> torch.Tensor:
    """Update Q-value using PyTorch."""
    return q_val + alpha * td_error

def batch_update_q_values(q_tensor: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, 
                         rewards: torch.Tensor, next_states: torch.Tensor, next_actions: torch.Tensor, 
                         dones: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Perform batch updates on Q-values using PyTorch."""
    # Check if we should use Numba on CPU
    if USE_NUMBA and DEVICE.type == 'cpu':
        # Convert tensors to numpy arrays for numba processing
        q_array = q_tensor.numpy()
        states_np = states.numpy()
        actions_np = actions.numpy()
        rewards_np = rewards.numpy()
        next_states_np = next_states.numpy()
        next_actions_np = next_actions.numpy()
        dones_np = dones.numpy()
        
        # Use numba-optimized function
        result_np = batch_update_q_values_cpu(
            q_array, states_np, actions_np, rewards_np, 
            next_states_np, next_actions_np, dones_np, alpha, gamma
        )
        
        # Convert back to tensor
        return torch.tensor(result_np, device=DEVICE)
    
    # PyTorch GPU or CPU implementation
    result = q_tensor.clone()
    
    # Use PyTorch's indexing for batch operations
    batch_size = states.size(0)
    
    # Convert states and next_states to proper indices if they're not already
    if states.dim() > 1:
        # Extract the state coordinates for indexing
        state_indices = (states[:, 0], states[:, 1])
        next_state_indices = (next_states[:, 0], next_states[:, 1])
    else:
        # If states are already flattened indices
        state_indices = states
        next_state_indices = next_states
    
    # Get Q-values for current state-action pairs
    current_q_values = result[state_indices[0], state_indices[1], actions]
    
    # Calculate targets based on whether the state is terminal
    # For terminal states, target is just the reward
    # For non-terminal states, include discounted future rewards
    next_q_values = result[next_state_indices[0], next_state_indices[1], next_actions]
    targets = rewards + gamma * next_q_values * (~dones)
    
    # Calculate TD errors
    td_errors = targets - current_q_values
    
    # Update Q-values 
    result[state_indices[0], state_indices[1], actions] += alpha * td_errors
    
    return result

def epsilon_greedy_action(q_values: torch.Tensor, epsilon: float) -> int:
    """Select action using epsilon-greedy strategy with PyTorch."""
    if USE_NUMBA and DEVICE.type == 'cpu':
        # Use numba version for CPU
        return epsilon_greedy_action_cpu(q_values.numpy(), epsilon)
        
    # PyTorch implementation
    if torch.rand(1, device=q_values.device).item() < epsilon:
        return torch.randint(0, q_values.size(0), (1,), device=q_values.device).item()
    else:
        return torch.argmax(q_values).item()

def softmax_action(q_values: torch.Tensor, temperature: float) -> int:
    """Select action using softmax strategy with PyTorch."""
    if USE_NUMBA and DEVICE.type == 'cpu':
        # Use numba version for CPU
        return softmax_action_cpu(q_values.numpy(), temperature)
        
    # PyTorch implementation
    # Normalize Q-values to prevent overflow
    q_values = q_values - torch.max(q_values)
    exp_q = torch.exp(q_values / temperature)
    probs = exp_q / torch.sum(exp_q)
    
    # Sample from the probability distribution
    return torch.multinomial(probs, 1).item()

def calculate_q_values_batch(state_batch: torch.Tensor, q_table: torch.Tensor) -> torch.Tensor:
    """Efficiently calculate Q-values for a batch of states using PyTorch."""
    return q_table[state_batch[:, 0], state_batch[:, 1], :]

def heuristic(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, 
              q_tensor: torch.Tensor, w1: float, w2: float) -> torch.Tensor:
    """
    Compute a heuristic value for A* search combining distance and Q-values using PyTorch.
    """
    # Manhattan distance
    distance = torch.abs(x2 - x1) + torch.abs(y2 - y1)
    
    # Extract Q-values for the current position
    q_values = q_tensor[x1, y1]
    max_q = torch.max(q_values, dim=-1)[0]
    
    # Q-value component (inverse, since higher Q-values are better)
    q_component = 1.0 / (1.0 + max_q)
    
    # Combine the two components with weights
    return w1 * q_component + w2 * distance

@njit
def calculate_td_error_cpu(q_val: float, next_q_val: float, reward: float, gamma: float) -> float:
    """
    Calculate TD error using Numba optimization.
    """
    return reward + gamma * next_q_val - q_val

@njit
def update_q_value_cpu(q_val: float, td_error: float, alpha: float) -> float:
    """Update Q-value using Numba optimization."""
    return q_val + alpha * td_error

@njit(parallel=True)
def batch_update_q_values_cpu(q_array: np.ndarray, states: np.ndarray, actions: np.ndarray, 
                          rewards: np.ndarray, next_states: np.ndarray, next_actions: np.ndarray, 
                          dones: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Perform batch updates on Q-values using Numba parallel optimization."""
    result = np.copy(q_array)
    for i in prange(len(states)):
        if dones[i]:
            target = rewards[i]
        else:
            target = rewards[i] + gamma * q_array[next_states[i][0], next_states[i][1], next_actions[i]]
        
        td_error = target - q_array[states[i][0], states[i][1], actions[i]]
        result[states[i][0], states[i][1], actions[i]] += alpha * td_error
    
    return result

@njit
def epsilon_greedy_action_cpu(q_values: np.ndarray, epsilon: float) -> int:
    """Select action using epsilon-greedy strategy with Numba optimization."""
    if np.random.random() < epsilon:
        return np.random.randint(0, len(q_values))
    else:
        return np.argmax(q_values)

@njit
def softmax_action_cpu(q_values: np.ndarray, temperature: float) -> int:
    """Select action using softmax strategy with Numba optimization."""
    # Normalize Q-values to prevent overflow
    q_values = q_values - np.max(q_values)
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_values), p=probs)

class ExperienceBuffer:
    """Thread-safe experience replay buffer."""
    def __init__(self, max_size: int = 10000, num_workers: int = 4):
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
        # Create individual buffers for each worker to reduce lock contention
        self.worker_buffers = [deque(maxlen=max_size // num_workers) for _ in range(num_workers)]
        self.num_workers = num_workers
    
    def add(self, experience: Experience, worker_id: int = None):
        """Add experience to buffer."""
        if worker_id is not None and 0 <= worker_id < self.num_workers:
            # Add to worker-specific buffer without locking
            self.worker_buffers[worker_id].append(experience)
        else:
            # Add to main buffer with locking
            with self._lock:
                self.buffer.append(experience)
    
    def merge_worker_buffers(self):
        """Merge worker-specific buffers into the main buffer."""
        with self._lock:
            for worker_buffer in self.worker_buffers:
                self.buffer.extend(worker_buffer)
                worker_buffer.clear()
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        # Ensure worker buffers are merged before sampling
        self.merge_worker_buffers()
        with self._lock:
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        # Count experiences in all buffers
        with self._lock:
            total_count = len(self.buffer)
            for worker_buffer in self.worker_buffers:
                total_count += len(worker_buffer)
            return total_count

class QAgent:
    """
    Individual agent implementing double Q-learning and responsible for a specific subtask.
    """
    def __init__(self, config: AgentConfig, system_config: HierarchicalAgentSystemConfig):
        self.config = config
        self.system_config = system_config
        self.id = config.agent_id
        
        # Initialize Q-tables as PyTorch tensors
        self.q_table = torch.zeros((system_config.grid_size[0], system_config.grid_size[1], 
                                 system_config.action_space), dtype=torch.float32, device=DEVICE)
        self.target_q_table = torch.zeros_like(self.q_table)  # Target network for double Q-learning
        
        # Initialize experience buffer with worker count matching learning threads
        self.experience_buffer = ExperienceBuffer(
            max_size=config.q_learning_config.replay_buffer_size,
            num_workers=system_config.max_learning_threads
        )
        
        # Set up communication channels
        self.message_queue = Queue()
        self.experiences_to_share = Queue()
        
        # Learning parameters
        self.alpha = config.q_learning_config.alpha
        self.gamma = config.q_learning_config.gamma
        self.epsilon = config.q_learning_config.epsilon
        self.epsilon_min = config.q_learning_config.epsilon_min
        self.epsilon_decay = config.q_learning_config.epsilon_decay
        
        # Setup thread locks
        self._q_lock = threading.RLock()
        self._param_lock = threading.RLock()
        self._episode_counter = 0
        self._step_counter = 0
        
        logger.info(f"Agent {self.id} initialized with subtask domain: {config.subtask_domain}")
    
    def select_action(self, state: Tuple[int, int]) -> int:
        """
        Select an action using the specified exploration strategy.
        """
        with self._q_lock:
            # Convert state to tensor indices and get q values
            q_values = self.q_table[state[0], state[1], :]
        
        if self.config.q_learning_config.exploration_strategy == "epsilon_greedy":
            return epsilon_greedy_action(q_values, self.epsilon)
        elif self.config.q_learning_config.exploration_strategy == "softmax":
            return softmax_action(q_values, self.config.temperature)
        else:
            # Default to epsilon-greedy
            return epsilon_greedy_action(q_values, self.epsilon)
    
    def update_q(self, experience: Experience):
        """
        Update Q-values using double Q-learning.
        """
        state, action, reward, next_state, done = (
            experience.state, experience.action, experience.reward, 
            experience.next_state, experience.done
        )
        
        # Convert to tensors
        state_tensor = (state[0], state[1])
        next_state_tensor = (next_state[0], next_state[1])
        reward_tensor = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
        done_tensor = torch.tensor(done, device=DEVICE, dtype=torch.bool)
        
        with self._q_lock:
            if self.config.use_double_q:
                # Double Q-learning: use primary network to select action, target network to evaluate
                next_action = torch.argmax(self.q_table[next_state_tensor[0], next_state_tensor[1], :]).item()
                next_q_val = self.target_q_table[next_state_tensor[0], next_state_tensor[1], next_action]
            else:
                # Standard Q-learning
                next_q_val = torch.max(self.q_table[next_state_tensor[0], next_state_tensor[1], :])
            
            current_q = self.q_table[state_tensor[0], state_tensor[1], action]
            
            if done:
                target = reward_tensor
            else:
                target = reward_tensor + self.gamma * next_q_val
            
            td_error = target - current_q
            self.q_table[state_tensor[0], state_tensor[1], action] += self.alpha * td_error
    
    def process_experiences(self, experiences: List[Experience], worker_id: int = None):
        """
        Process a batch of experiences for Q-learning updates.
        """
        if not experiences:
            return
        
        # Add experiences to buffer
        for exp in experiences:
            self.experience_buffer.add(exp, worker_id)
        
        # Sample a batch if buffer is large enough
        if len(self.experience_buffer) >= self.config.q_learning_config.batch_size:
            self._batch_update()
    
    def _batch_update(self):
        """
        Perform batch updates on Q-values using experiences from the buffer.
        """
        batch = self.experience_buffer.sample(self.config.q_learning_config.batch_size)
        
        # Prepare tensors for batch update
        states = torch.tensor([[exp.state[0], exp.state[1]] for exp in batch], 
                             dtype=torch.long, device=DEVICE)
        actions = torch.tensor([exp.action for exp in batch], 
                              dtype=torch.long, device=DEVICE)
        rewards = torch.tensor([exp.reward for exp in batch], 
                              dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor([[exp.next_state[0], exp.next_state[1]] for exp in batch], 
                                  dtype=torch.long, device=DEVICE)
        dones = torch.tensor([exp.done for exp in batch], 
                            dtype=torch.bool, device=DEVICE)
        
        with self._q_lock:
            if self.config.use_double_q:
                # Double Q-learning batch update
                # Get best actions from primary network
                next_actions = torch.argmax(
                    self.q_table[next_states[:, 0], next_states[:, 1], :], dim=1
                )
                
                # Use PyTorch-based batch update
                self.q_table = batch_update_q_values(
                    self.q_table, states, actions, rewards, 
                    next_states, next_actions, dones, self.alpha, self.gamma
                )
            else:
                # Get best actions for next states
                next_actions = torch.argmax(
                    self.q_table[next_states[:, 0], next_states[:, 1], :], dim=1
                )
                
                # Use PyTorch-based batch update
                self.q_table = batch_update_q_values(
                    self.q_table, states, actions, rewards, 
                    next_states, next_actions, dones, self.alpha, self.gamma
                )
    
    def update_target_network(self):
        """
        Update the target network with the current Q-table values.
        """
        with self._q_lock:
            self.target_q_table = self.q_table.clone()
    
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        """
        with self._param_lock:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def receive_message(self, message: Dict[str, Any]):
        """
        Process a message from another agent.
        """
        self.message_queue.put(message)
    
    def process_messages(self):
        """
        Process all messages in the message queue.
        """
        while not self.message_queue.empty():
            message = self.message_queue.get()
            message_type = message.get('type')
            
            if message_type == 'q_values':
                self._process_q_values_message(message)
            elif message_type == 'experience':
                self._process_experience_message(message)
            elif message_type == 'coordination':
                self._process_coordination_message(message)
    
    def _process_q_values_message(self, message: Dict[str, Any]):
        """
        Process a message containing Q-values from another agent.
        """
        sender_id = message.get('sender_id')
        q_values = message.get('q_values')
        states = message.get('states')
        
        with self._q_lock:
            for state, q_vals in zip(states, q_values):
                # Only update if the state is in our domain
                if self._is_in_domain(state):
                    # Weighted average of our Q-values and received Q-values
                    weight = self.config.heuristic_weight
                    q_vals_tensor = torch.tensor(q_vals, device=DEVICE, dtype=torch.float32)
                    
                    # Compute weighted average
                    self.q_table[state[0], state[1], :] = (
                        (1 - weight) * self.q_table[state[0], state[1], :] + 
                        weight * q_vals_tensor
                    )
    
    def _process_experience_message(self, message: Dict[str, Any]):
        """
        Process a message containing shared experiences from another agent.
        """
        sender_id = message.get('sender_id')
        experiences = message.get('experiences')
        
        # Filter experiences that are in our domain
        relevant_experiences = [
            Experience(**exp) for exp in experiences 
            if self._is_in_domain(exp['state']) or self._is_in_domain(exp['next_state'])
        ]
        
        self.process_experiences(relevant_experiences)
    
    def _process_coordination_message(self, message: Dict[str, Any]):
        """
        Process a coordination message from another agent or the coordinator.
        """
        coordination_type = message.get('coordination_type')
        
        if coordination_type == 'task_reassignment':
            # Update subtask domain
            new_domain = message.get('new_domain')
            if new_domain:
                self.config.subtask_domain = new_domain
                logger.info(f"Agent {self.id} domain reassigned to: {new_domain}")
        elif coordination_type == 'parameter_adjustment':
            # Update learning parameters
            params = message.get('parameters', {})
            with self._param_lock:
                for param, value in params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                        logger.info(f"Agent {self.id} updated parameter {param} to {value}")
    
    def share_experiences(self) -> List[Dict]:
        """
        Share valuable experiences with other agents.
        Returns experiences as serializable dictionaries.
        """
        if len(self.experience_buffer) < self.config.q_learning_config.batch_size:
            return []
        
        # Sample high-reward experiences to share
        experiences = self.experience_buffer.sample(self.config.q_learning_config.batch_size // 2)
        threshold = self.system_config.experience_sharing_threshold
        
        # Filter experiences with rewards above threshold
        valuable_experiences = [exp for exp in experiences if exp.reward > threshold]
        
        # Convert to dictionaries for sharing
        return [exp.dict() for exp in valuable_experiences]
    
    def share_q_values(self, states: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Share Q-values for specific states with other agents.
        """
        q_values = []
        with self._q_lock:
            for state in states:
                if self._is_in_domain(state):
                    # Convert tensor to list for serialization
                    q_values.append(self.q_table[state[0], state[1], :].cpu().tolist())
        
        return {
            'sender_id': self.id,
            'type': 'q_values',
            'states': states,
            'q_values': q_values
        }
    
    def _is_in_domain(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is in this agent's domain.
        """
        if not self.config.subtask_domain:
            return True  # If no domain specified, consider all states in domain
        
        return state in self.config.subtask_domain
    
    def incorporate_pathfinder_data(self, path: List[Tuple[int, int]], goal: Tuple[int, int], reward: float):
        """
        Incorporate path data from A* pathfinding to guide Q-learning.
        """
        if not path or len(path) < 2:
            return
        
        # Create synthetic experiences from the path
        for i in range(len(path) - 1):
            current = path[i]
            next_state = path[i + 1]
            
            # Determine action that led from current to next_state
            dx = next_state[0] - current[0]
            dy = next_state[1] - current[1]
            
            # Map (dx, dy) to action index based on DIRECTIONS_8 from pathfinder
            # [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            actions_map = {
                (1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3,
                (1, 1): 4, (1, -1): 5, (-1, 1): 6, (-1, -1): 7
            }
            
            action = actions_map.get((dx, dy), 0)
            
            # Calculate immediate reward (higher for closer to goal)
            done = (next_state == goal)
            immediate_reward = reward / len(path) if not done else reward
            
            # Create and add experience
            exp = Experience(
                state=current,
                action=action,
                reward=immediate_reward,
                next_state=next_state,
                done=done
            )
            
            self.experience_buffer.add(exp)
    
    def track_episode(self):
        """
        Track episode progress and trigger periodic actions.
        """
        with self._param_lock:
            self._episode_counter += 1
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Update target network periodically
            if self.config.use_double_q and self._episode_counter % self.config.target_update_frequency == 0:
                self.update_target_network()

class HierarchicalAgentSystem:
    """
    Manages a system of hierarchical agents that work together to solve complex tasks.
    """
    def __init__(self, config: HierarchicalAgentSystemConfig):
        self.config = config
        self.agents = {}
        
        # Initialize cache and resource managers for pathfinding
        self.cache_config = CacheConfig(
            use_redis=self.config.agents_config[0].q_learning_config.cache_storage == CacheStorage.REDIS,
            max_connections=100,  # Fixed value instead of CacheConfig.max_connections
            cache_maxsize=10000,  # Increased for better performance
            cache_ttl=600.0  # 10 minutes TTL
        )
        
        self.resource_config = ResourceConfig(
            enabled=True,
            observation_period=5.0,
            check_interval=0.5,
            target_utilization=0.85
        )
        
        # Initialize agents
        for agent_config in config.agents_config:
            self.agents[agent_config.agent_id] = QAgent(agent_config, config)
        
        # Initialize pathfinder
        self.pathfinder_config = config.pathfinder_config or PathfinderConfig(
            grid_size=config.grid_size,
            allow_diagonal=(config.action_space == 8),
            concurrent_expansions=min(100, config.max_communication_threads * 2)  # Increased concurrency
        )
        
        # Communication and coordination
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_communication_threads,
            thread_name_prefix="comm_thread"
        )
        self._learning_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_learning_threads,
            thread_name_prefix="learn_thread"
        )
        
        # System state tracking
        self._episode_counter = 0
        self._system_lock = threading.RLock()
        
        # Create domain partitioning if not specified
        self._ensure_domain_partitioning()
        
        logger.info(f"Hierarchical Agent System initialized with {len(self.agents)} agents on device: {DEVICE}")
    
    def _ensure_domain_partitioning(self):
        """
        Ensure that each agent has a defined domain by partitioning the state space.
        """
        # Check if domains are already assigned
        all_assigned = all(bool(agent.config.subtask_domain) for agent in self.agents.values())
        if all_assigned:
            return
        
        # Simple grid partitioning
        grid_width, grid_height = self.config.grid_size
        num_agents = len(self.agents)
        
        # Determine partitioning strategy (try to make square-ish partitions)
        partitions_x = int(np.ceil(np.sqrt(num_agents)))
        partitions_y = int(np.ceil(num_agents / partitions_x))
        
        width_per_partition = grid_width // partitions_x
        height_per_partition = grid_height // partitions_y
        
        # Assign domains to agents
        agent_list = list(self.agents.values())
        partition_idx = 0
        
        for i in range(partitions_x):
            for j in range(partitions_y):
                if partition_idx < num_agents:
                    x_start = i * width_per_partition
                    y_start = j * height_per_partition
                    x_end = min((i + 1) * width_per_partition, grid_width)
                    y_end = min((j + 1) * height_per_partition, grid_height)
                    
                    # Create domain as list of coordinates
                    domain = []
                    for x in range(x_start, x_end):
                        for y in range(y_start, y_end):
                            domain.append((x, y))
                    
                    agent_list[partition_idx].config.subtask_domain = domain
                    logger.info(f"Assigned domain to agent {agent_list[partition_idx].id}: {x_start}:{x_end}, {y_start}:{y_end}")
                    
                    partition_idx += 1
    
    async def train(self, environment, num_episodes: int, q_value_map: Optional[Dict] = None):
        """
        Train the multi-agent system on the given environment.
        
        Args:
            environment: Environment that supports reset() and step(action) methods
            num_episodes: Number of episodes to train for
            q_value_map: Initial Q-values to guide exploration (optional)
        """
        pathfinder = None
        if q_value_map:
            # Initialize pathfinder with Q-values for heuristic guidance
            pathfinder = AStarPathfinder(
                q_values=q_value_map,
                config=self.pathfinder_config
            )
        
        for episode in range(num_episodes):
            await self._run_episode(environment, episode, pathfinder)
            
            # Coordinate agents periodically
            if episode % self.config.coordination_frequency == 0:
                await self._coordinate_agents()
            
            self._episode_counter = episode
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Completed episode {episode}/{num_episodes}")
        
        logger.info(f"Training completed after {num_episodes} episodes")
    
    async def _run_episode(self, environment, episode: int, pathfinder: Optional[AStarPathfinder] = None):
        """
        Run a single training episode with all agents participating.
        """
        try:
            state = environment.reset()
            done = False
            total_reward = 0
            
            # Determine goal state for this episode (assuming environment has a goal)
            goal = getattr(environment, 'goal', None)
            
            # Use A* pathfinder to find optimal path if available
            optimal_path = None
            if pathfinder and goal:
                try:
                    cache_manager = CacheManager(config=self.cache_config)
                    resource_manager = ResourceManager(config=self.resource_config)
                    
                    # Initialize pathfinder with properly configured dependencies
                    enhanced_pathfinder = AStarPathfinder(
                        q_values=self._get_q_values_dict(), 
                        config=self.pathfinder_config,
                        cache_config=self.cache_config,
                        cache_manager=cache_manager,
                        resource_manager=resource_manager
                    )
                    
                    optimal_path, _ = await enhanced_pathfinder.bidirectional_a_star(state, goal)
                    
                    # Share optimal path with agents for guided exploration
                    if optimal_path:
                        path_reward = 1.0  # Reward for following optimal path
                        
                        # Use a more efficient approach to incorporate path data
                        worker_tasks = []
                        for i, agent in enumerate(self.agents.values()):
                            worker_id = i % self.config.max_learning_threads
                            worker_tasks.append(
                                self._learning_executor.submit(
                                    agent.incorporate_pathfinder_data, 
                                    optimal_path, goal, path_reward
                                )
                            )
                        
                        # Wait for all tasks to complete
                        for task in concurrent.futures.as_completed(worker_tasks):
                            # Handle any exceptions
                            try:
                                task.result()
                            except Exception as e:
                                logger.error(f"Error in path incorporation task: {e}")
                except Exception as e:
                    logger.error(f"Error in pathfinding: {e}")
                    # Continue without pathfinder guidance
            
            # Track steps in this episode
            step = 0
            max_steps = self.config.agents_config[0].q_learning_config.max_steps
            
            while not done and step < max_steps:
                try:
                    # Get responsible agent for current state
                    responsible_agent = self._get_responsible_agent(state)
                    
                    # Let the agent select an action
                    action = responsible_agent.select_action(state)
                    
                    # Take action in environment
                    next_state, reward, done, info = environment.step(action)
                    
                    # Create experience
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )
                    
                    # Update the responsible agent
                    await self._learning_executor.submit(responsible_agent.process_experiences, [experience])
                    
                    # Share valuable experience with other agents if reward is significant
                    if reward > self.config.experience_sharing_threshold:
                        await self._share_experience(experience)
                    
                    state = next_state
                    total_reward += reward
                    step += 1
                    
                    # Process messages periodically
                    if step % 5 == 0:
                        await self._process_all_messages()
                except Exception as e:
                    logger.error(f"Error during episode step {step}: {e}")
                    # Try to continue with the next step
            
            # Track episode completion for all agents
            try:
                await asyncio.gather(*[
                    self._learning_executor.submit(agent.track_episode)
                    for agent in self.agents.values()
                ])
            except Exception as e:
                logger.error(f"Error tracking episode completion: {e}")
            
            logger.debug(f"Episode {episode} completed with reward {total_reward}")
            return total_reward
        except Exception as e:
            logger.error(f"Error running episode {episode}: {e}")
            return 0  # Return zero reward on failure
    
    def _get_responsible_agent(self, state: Tuple[int, int]) -> QAgent:
        """
        Determine which agent is responsible for the given state.
        """
        for agent in self.agents.values():
            if agent._is_in_domain(state):
                return agent
        
        # If no agent claims the state, return the first agent as default
        logger.warning(f"No agent responsible for state {state}, using default")
        return next(iter(self.agents.values()))
    
    async def _share_experience(self, experience: Experience):
        """
        Share a valuable experience with all agents.
        """
        for agent in self.agents.values():
            # Skip the experience if it's not in or adjacent to the agent's domain
            if not (agent._is_in_domain(experience.state) or agent._is_in_domain(experience.next_state)):
                continue
            
            # Share the experience
            message = {
                'type': 'experience',
                'sender_id': 'coordinator',
                'experiences': [experience.dict()]
            }
            await self._thread_executor.submit(agent.receive_message, message)
    
    async def _process_all_messages(self):
        """
        Process all pending messages for all agents.
        """
        await asyncio.gather(*[
            self._thread_executor.submit(agent.process_messages)
            for agent in self.agents.values()
        ])
    
    async def _coordinate_agents(self):
        """
        Perform coordination activities between agents.
        """
        # Share Q-values between agents for overlapping areas
        await self._share_q_values()
        
        # Check for load balancing opportunities
        await self._balance_agent_loads()
    
    async def _share_q_values(self):
        """
        Share Q-values between agents for overlapping or boundary states.
        """
        # Identify boundary states between agents
        boundary_states = self._identify_boundary_states()
        
        # Have each agent share its Q-values for boundary states
        for agent_id, states in boundary_states.items():
            agent = self.agents[agent_id]
            q_values_message = await self._thread_executor.submit(agent.share_q_values, states)
            
            # Share with relevant agents
            for other_agent in self.agents.values():
                if other_agent.id != agent_id:
                    await self._thread_executor.submit(other_agent.receive_message, q_values_message)
    
    def _identify_boundary_states(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Identify boundary states between agent domains.
        """
        boundary_states = {agent_id: [] for agent_id in self.agents.keys()}
        
        # For each agent, find states that are at the boundary of its domain
        for agent_id, agent in self.agents.items():
            if not agent.config.subtask_domain:
                continue
                
            # For each state in the domain
            for state in agent.config.subtask_domain:
                x, y = state
                
                # Check neighboring states
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        neighbor = (x + dx, y + dy)
                        
                        # If neighbor is valid but not in domain, this is a boundary
                        if (0 <= neighbor[0] < self.config.grid_size[0] and 
                            0 <= neighbor[1] < self.config.grid_size[1] and 
                            neighbor not in agent.config.subtask_domain):
                            
                            boundary_states[agent_id].append(state)
                            break
        
        return boundary_states
    
    async def _balance_agent_loads(self):
        """
        Balance the workload between agents by reassigning parts of domains.
        """
        # Calculate the load on each agent (could be based on domain size, visit frequency, etc.)
        agent_loads = {}
        for agent_id, agent in self.agents.items():
            # Simple metric: domain size
            agent_loads[agent_id] = len(agent.config.subtask_domain) if agent.config.subtask_domain else 0
        
        # Find imbalances
        avg_load = sum(agent_loads.values()) / len(agent_loads)
        overloaded = []
        underloaded = []
        
        for agent_id, load in agent_loads.items():
            # Consider an agent overloaded/underloaded if 20% above/below average
            if load > avg_load * 1.2:
                overloaded.append((agent_id, load))
            elif load < avg_load * 0.8 and load > 0:  # Ensure agent has some load
                underloaded.append((agent_id, load))
        
        # Balance by transferring states from overloaded to underloaded agents
        if overloaded and underloaded:
            # Sort by load (descending for overloaded, ascending for underloaded)
            overloaded.sort(key=lambda x: x[1], reverse=True)
            underloaded.sort(key=lambda x: x[1])
            
            for over_id, _ in overloaded:
                for under_id, _ in underloaded:
                    over_agent = self.agents[over_id]
                    under_agent = self.agents[under_id]
                    
                    # Find transferable states (boundary states in overloaded agent's domain)
                    transferable = self._find_transferable_states(over_agent, under_agent)
                    
                    if transferable:
                        # Transfer a portion of states
                        transfer_count = min(len(transferable) // 4, 
                                        int((agent_loads[over_id] - agent_loads[under_id]) // 2))
                        transfer_count = max(1, transfer_count)  # Transfer at least one state
                        
                        states_to_transfer = transferable[:transfer_count]
                        
                        # Update domains
                        over_agent.config.subtask_domain = [
                            state for state in over_agent.config.subtask_domain 
                            if state not in states_to_transfer
                        ]
                        under_agent.config.subtask_domain.extend(states_to_transfer)
                        
                        # Notify agents of the change
                        await self._notify_domain_change(over_id, under_id, states_to_transfer)
                        
                        logger.info(f"Transferred {len(states_to_transfer)} states from agent {over_id} to {under_id}")
                        
                        # Update loads for next iteration
                        agent_loads[over_id] -= len(states_to_transfer)
                        agent_loads[under_id] += len(states_to_transfer)
    
    def _find_transferable_states(self, over_agent: QAgent, under_agent: QAgent) -> List[Tuple[int, int]]:
        """
        Find states that can be transferred from overloaded to underloaded agent.
        """
        if not over_agent.config.subtask_domain:
            return []
        
        # Find boundary states in overloaded agent's domain that are adjacent to underloaded agent
        transferable = []
        under_domain = set(under_agent.config.subtask_domain) if under_agent.config.subtask_domain else set()
        
        for state in over_agent.config.subtask_domain:
            x, y = state
            
            # Check if this state is at the boundary
            is_boundary = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (x + dx, y + dy)
                    if neighbor in under_domain:
                        is_boundary = True
                        break
                
                if is_boundary:
                    break
            
            if is_boundary:
                transferable.append(state)
        
        return transferable
    
    async def _notify_domain_change(self, from_agent_id: str, to_agent_id: str, states: List[Tuple[int, int]]):
        """
        Notify agents of domain changes.
        """
        # Notify the agent losing domain
        message_from = {
            'type': 'coordination',
            'coordination_type': 'task_reassignment',
            'new_domain': self.agents[from_agent_id].config.subtask_domain
        }
        await self._thread_executor.submit(self.agents[from_agent_id].receive_message, message_from)
        
        # Notify the agent gaining domain
        message_to = {
            'type': 'coordination',
            'coordination_type': 'task_reassignment',
            'new_domain': self.agents[to_agent_id].config.subtask_domain
        }
        await self._thread_executor.submit(self.agents[to_agent_id].receive_message, message_to)
    
    def get_consolidated_q_table(self) -> torch.Tensor:
        """
        Returns a consolidated view of all agent Q-tables.
        Combines Q-tables by taking max values where domains overlap.
        """
        # Initialize with first agent if available
        if not self.agents:
            # Return empty tensor with same device as module is using
            return torch.zeros((1, 1, 1), device=DEVICE)
        
        # Get first agent's Q-table for shape reference
        first_agent_id = next(iter(self.agents))
        first_agent = self.agents[first_agent_id]
        consolidated = first_agent.q_table.clone()  # Create a copy to avoid modifying original
        
        # Merge with other agents
        for agent_id, agent in self.agents.items():
            if agent_id == first_agent_id:
                continue
            
            # Handle different Q-table shapes (different action spaces)
            if agent.q_table.shape[2] != consolidated.shape[2]:
                # If different action dimensions, pad the smaller one
                smaller_dim = min(agent.q_table.shape[2], consolidated.shape[2])
                # Take maximum values for common actions
                consolidated[:, :, :smaller_dim] = torch.maximum(
                    consolidated[:, :, :smaller_dim], 
                    agent.q_table[:, :, :smaller_dim]
                )
            else:
                # Same action dimensions, simply take element-wise maximum
                consolidated = torch.maximum(consolidated, agent.q_table)
        
        return consolidated
    
    def save_q_tables(self, save_path: str):
        """
        Save all agent Q-tables to disk.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save individual agent Q-tables
        for agent_id, agent in self.agents.items():
            agent_path = os.path.join(save_path, f"agent_{agent_id}_q_table.pt")
            # Move tensor to CPU before saving
            q_table_cpu = agent.q_table.cpu()
            torch.save(q_table_cpu, agent_path)
        
        # Save consolidated Q-table
        consolidated_path = os.path.join(save_path, "consolidated_q_table.pt")
        consolidated_q = self.get_consolidated_q_table().cpu()
        torch.save(consolidated_q, consolidated_path)
        
        logger.info(f"Saved Q-tables to {save_path}")
    
    def load_q_tables(self, load_path: str):
        """
        Load agent Q-tables from disk.
        """
        # Load individual agent Q-tables
        for agent_id, agent in self.agents.items():
            agent_path = os.path.join(load_path, f"agent_{agent_id}_q_table.pt")
            if os.path.exists(agent_path):
                with agent._q_lock:
                    # Load to the appropriate device with weights_only=True for security
                    agent.q_table = torch.load(agent_path, map_location=DEVICE)
                    # Copy to target network if using double Q-learning
                    if agent.config.use_double_q:
                        agent.target_q_table = agent.q_table.clone()
                    logger.info(f"Loaded Q-table for agent {agent_id}")
    
    def _get_q_values_dict(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Convert the consolidated Q-table to the format expected by the pathfinder.
        """
        q_values = {}
        grid_size = self.config.grid_size
        action_space = self.config.action_space
        
        # For each state, create a dictionary of action values
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                state = (x, y)
                agent = self._get_responsible_agent(state)
                
                # Get Q-values for this state
                with agent._q_lock:
                    # Get values as CPU numpy array
                    state_q_values = agent.q_table[x, y, :].cpu().numpy()
                
                # Convert to dictionary format for pathfinder
                action_values = {}
                for a in range(action_space):
                    action_values[f"action_{a}"] = float(state_q_values[a])
                
                q_values[state] = action_values
        
        return q_values

    async def select_action(self, state: Tuple[int, int]) -> int:
        """
        Select an action for the given state using the responsible agent.
        """
        agent = self._get_responsible_agent(state)
        return agent.select_action(state)
    
    def close(self):
        """
        Clean up resources.
        """
        self._thread_executor.shutdown()
        self._learning_executor.shutdown()

class HierarchicalMultiAgentEnvironment:
    """
    A wrapper environment that supports the hierarchical multi-agent system.
    This environment can be customized depending on the specific problem domain.
    """
    def __init__(self, grid_size: Tuple[int, int], obstacles: List[Tuple[int, int]] = None):
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.agent_pos = (0, 0)
        self.goal = (grid_size[0] - 1, grid_size[1] - 1)
        self.done = False
        self.max_steps = grid_size[0] * grid_size[1]  # A reasonable upper bound
        self.current_step = 0
        
        # Create tensor versions for faster calculations
        self.agent_pos_tensor = torch.tensor(self.agent_pos, dtype=torch.long, device=DEVICE)
        self.goal_tensor = torch.tensor(self.goal, dtype=torch.long, device=DEVICE)
        if obstacles:
            self.obstacles_tensor = torch.tensor(obstacles, dtype=torch.long, device=DEVICE)
        else:
            self.obstacles_tensor = torch.zeros((0, 2), dtype=torch.long, device=DEVICE)
    
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to initial state.
        """
        self.agent_pos = (0, 0)
        self.agent_pos_tensor = torch.tensor(self.agent_pos, dtype=torch.long, device=DEVICE)
        self.done = False
        self.current_step = 0
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: Integer representing the action (0-7 for 8-directional movement)
            
        Returns:
            next_state: The new agent position
            reward: The reward for this step
            done: Whether the episode is done
            info: Additional information
        """
        self.current_step += 1
        
        # Map action to direction
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        
        if action < len(directions):
            dx, dy = directions[action]
            next_x = self.agent_pos[0] + dx
            next_y = self.agent_pos[1] + dy
            
            # Check bounds
            if (0 <= next_x < self.grid_size[0] and 
                0 <= next_y < self.grid_size[1]):
                
                # Check if next position is an obstacle
                next_pos = (next_x, next_y)
                is_obstacle = False
                
                if len(self.obstacles) > 0:
                    # Convert to tensor for faster comparison
                    next_pos_tensor = torch.tensor(next_pos, dtype=torch.long, device=DEVICE)
                    # Check if next position matches any obstacle
                    is_obstacle = torch.any(torch.all(self.obstacles_tensor == next_pos_tensor, dim=1)).item()
                
                if not is_obstacle:
                    self.agent_pos = next_pos
                    self.agent_pos_tensor = torch.tensor(self.agent_pos, dtype=torch.long, device=DEVICE)
        
        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 10.0
            self.done = True
        else:
            # Calculate distance using tensors for performance
            dist_tensor = torch.sqrt(torch.sum(torch.square(
                self.agent_pos_tensor.float() - self.goal_tensor.float()
            ))).item()
            reward = -0.1 - 0.01 * dist_tensor
        
        # Check if we've reached maximum steps
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self.agent_pos, reward, self.done, {}
    
    def render(self):
        """
        Render the environment (text-based for simplicity).
        """
        grid = []
        for y in range(self.grid_size[1]):
            row = []
            for x in range(self.grid_size[0]):
                if (x, y) == self.agent_pos:
                    row.append('A')
                elif (x, y) == self.goal:
                    row.append('G')
                elif (x, y) in self.obstacles:
                    row.append('#')
                else:
                    row.append('.')
            grid.append(''.join(row))
        
        return '\n'.join(grid)

# Deep Q-Network model using PyTorch for more complex environments
class DQNModel(torch.nn.Module):
    """
    Deep Q-Network model for more complex environments beyond simple table lookup.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super(DQNModel, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

class DQNConfig(BaseModel):
    """Configuration for DQN extension"""
    use_dqn: bool = Field(
        default=False,
        description="Whether to use DQN instead of Q-table"
    )
    feature_dim: int = Field(
        default=10,
        description="Dimension of feature vector for DQN input"
    )
    hidden_dims: List[int] = Field(
        default=[128, 64],
        description="Hidden layer dimensions"
    )
    target_update_frequency: int = Field(
        default=100,
        description="How often to update target network"
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate for optimizer"
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for DQN training"
    )
    buffer_size: int = Field(
        default=10000,
        description="Size of experience replay buffer"
    )

class DQNAgent(QAgent):
    """
    Extension of QAgent that uses Deep Q-Networks instead of Q-tables.
    Allows for more complex state representations and environments.
    """
    def __init__(self, config: AgentConfig, system_config: HierarchicalAgentSystemConfig, dqn_config: DQNConfig):
        super().__init__(config, system_config)
        
        self.dqn_config = dqn_config
        
        # Feature dimension and action space
        self.feature_dim = dqn_config.feature_dim
        self.action_space = system_config.action_space
        
        # Initialize DQN models
        self.policy_net = DQNModel(
            input_dim=self.feature_dim,
            output_dim=self.action_space,
            hidden_dims=dqn_config.hidden_dims
        ).to(DEVICE)
        
        self.target_net = DQNModel(
            input_dim=self.feature_dim,
            output_dim=self.action_space,
            hidden_dims=dqn_config.hidden_dims
        ).to(DEVICE)
        
        # Copy weights from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Set target network to evaluation mode
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=dqn_config.learning_rate
        )
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Training step counter
        self.training_steps = 0
        
        logger.info(f"DQN Agent {self.id} initialized with feature dimension {self.feature_dim}")
    
    def state_to_features(self, state: Tuple[int, int]) -> torch.Tensor:
        """
        Convert a state to feature vector.
        This should be extended for more complex environments.
        """
        # Simple example: one-hot encoding of position
        x, y = state
        grid_size_x, grid_size_y = self.system_config.grid_size
        
        # Basic features: normalized position
        features = torch.zeros(self.feature_dim, device=DEVICE)
        features[0] = x / grid_size_x  # Normalized x position
        features[1] = y / grid_size_y  # Normalized y position
        
        # Additional features could be added here for more complex environments
        
        return features
    
    def select_action(self, state: Tuple[int, int]) -> int:
        """Select action using DQN"""
        features = self.state_to_features(state).unsqueeze(0)  # Add batch dimension
        
        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                # Forward pass through policy network
                q_values = self.policy_net(features)
                # Select action with highest Q-value
                return torch.argmax(q_values).item()
    
    def _batch_update(self):
        """Train the DQN on a batch of experiences"""
        if len(self.experience_buffer) < self.dqn_config.batch_size:
            return
        
        # Sample batch from experience buffer
        batch = self.experience_buffer.sample(self.dqn_config.batch_size)
        
        # Extract batch components
        states = [exp.state for exp in batch]
        actions = [exp.action for exp in batch]
        rewards = [exp.reward for exp in batch]
        next_states = [exp.next_state for exp in batch]
        dones = [exp.done for exp in batch]
        
        # Convert to feature tensors
        state_features = torch.stack([self.state_to_features(s) for s in states])
        next_state_features = torch.stack([self.state_to_features(s) for s in next_states])
        
        # Convert other components to tensors
        action_tensor = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=DEVICE)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_features).gather(1, action_tensor)
        
        # Compute target Q values
        with torch.no_grad():
            # Double DQN: use policy net to select actions, target net to evaluate
            next_actions = self.policy_net(next_state_features).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_features).gather(1, next_actions)
            # Set Q-values of terminal states to reward only
            target_q_values = reward_tensor.unsqueeze(1) + self.gamma * next_q_values * (~done_tensor).unsqueeze(1)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Increment step counter
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.dqn_config.target_update_frequency == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path: str):
        """Save DQN models to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)
        logger.info(f"DQN model saved to {path}")
    
    def load_model(self, path: str):
        """Load DQN models from disk"""
        if not os.path.exists(path):
            logger.warning(f"Model file {path} not found")
            return False
        
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        logger.info(f"DQN model loaded from {path}")
        return True

class MAMLQAgentSystem(HierarchicalAgentSystem):
    """
    Extended hierarchical agent system with MAML support.
    Implements Model-Agnostic Meta-Learning for Q-learning agents.
    """
    def __init__(self, config: HierarchicalAgentSystemConfig):
        super().__init__(config)
        # MAML-specific state tracking
        self.meta_parameters = {}  # Store meta-parameters (outer loop)
        self.task_parameters = {}  # Store task-specific parameters (inner loop)
        self.meta_optimizer_state = {}  # Store optimizer state for meta-updates
        
    async def meta_train(self, task_environments: List[Any], num_meta_iterations: int):
        """
        Perform MAML meta-training using the provided task environments.
        """
        for iteration in range(num_meta_iterations):
            # 1. Store current meta-parameters (the initial point for all tasks)
            self._store_meta_parameters()
            
            # 2. Sample batch of tasks
            batch_size = min(self.config.agents_config[0].q_learning_config.meta_batch_size, len(task_environments))
            task_batch = random.sample(task_environments, batch_size)
            
            task_gradients = []
            
            # 3. For each task, perform inner loop adaptation
            for task_env in task_batch:
                # 3.1 Reset to meta-parameters
                self._load_meta_parameters()
                
                # 3.2 Adapt to the specific task (inner loop)
                await self.adapt_to_task(task_env)
                
                # 3.3 Compute "gradient" as difference between adapted and initial parameters
                task_grad = self._compute_task_gradient()
                task_gradients.append(task_grad)
            
            # 4. Update meta-parameters (outer loop)
            self._update_meta_parameters(task_gradients)
            
            # Log meta-training progress
            if iteration % 10 == 0:
                logger.info(f"Completed meta-iteration {iteration}/{num_meta_iterations}")
    
    async def adapt_to_task(self, environment, num_episodes: Optional[int] = None):
        """
        Adapt the model to a specific task (inner loop of MAML).
        """
        episodes = num_episodes or self.config.agents_config[0].q_learning_config.adaptation_episodes
        adaptation_rate = self.config.agents_config[0].q_learning_config.adapt_lr
        
        # Store current learning rates and temporarily set to adaptation rate
        original_rates = {}
        for agent_id, agent in self.agents.items():
            original_rates[agent_id] = agent.alpha
            agent.alpha = adaptation_rate
        
        try:
            # Run adaptation episodes
            await self.train(environment, episodes)
            
            # Store adapted parameters
            self._store_task_parameters()
        finally:
            # Restore original learning rates
            for agent_id, agent in self.agents.items():
                agent.alpha = original_rates[agent_id]
    
    def _store_meta_parameters(self):
        """Store current parameters as meta-parameters"""
        try:
            self.meta_parameters = {}
            for agent_id, agent in self.agents.items():
                with agent._q_lock:
                    self.meta_parameters[agent_id] = agent.q_table.clone()
            logger.debug("Stored meta-parameters")
        except Exception as e:
            logger.error(f"Error storing meta-parameters: {e}")
    
    def _store_task_parameters(self):
        """Store current parameters as task-specific parameters"""
        try:
            self.task_parameters = {}
            for agent_id, agent in self.agents.items():
                with agent._q_lock:
                    self.task_parameters[agent_id] = agent.q_table.clone()
            logger.debug("Stored task-specific parameters")
        except Exception as e:
            logger.error(f"Error storing task parameters: {e}")
    
    def _load_meta_parameters(self):
        """Load stored meta-parameters into agents"""
        try:
            for agent_id, agent in self.agents.items():
                if agent_id in self.meta_parameters:
                    with agent._q_lock:
                        agent.q_table = self.meta_parameters[agent_id].clone()
                        if agent.config.use_double_q:
                            agent.target_q_table = self.meta_parameters[agent_id].clone()
            logger.debug("Loaded meta-parameters")
        except Exception as e:
            logger.error(f"Error loading meta-parameters: {e}")
    
    def _compute_task_gradient(self):
        """
        Compute the "gradient" as the difference between adapted parameters and meta-parameters.
        In Q-learning context, this is a simplified approximation of the gradient.
        """
        try:
            task_grad = {}
            for agent_id in self.agents.keys():
                if agent_id in self.meta_parameters and agent_id in self.task_parameters:
                    # "Gradient" is the difference between adapted and initial parameters
                    task_grad[agent_id] = self.task_parameters[agent_id] - self.meta_parameters[agent_id]
            return task_grad
        except Exception as e:
            logger.error(f"Error computing task gradient: {e}")
            return {}
    
    def _update_meta_parameters(self, task_gradients: List[Dict[str, torch.Tensor]]):
        """
        Update meta-parameters using the average gradient from all tasks.
        """
        try:
            if not task_gradients:
                return
            
            # Compute average gradient across tasks
            avg_grad = {}
            for agent_id in self.agents.keys():
                agent_grads = [g[agent_id] for g in task_gradients if agent_id in g]
                if agent_grads:
                    # Sum tensors and divide by count
                    avg_grad[agent_id] = torch.stack(agent_grads).mean(dim=0)
            
            # Update meta-parameters with average gradient
            meta_lr = self.config.agents_config[0].q_learning_config.meta_lr
            for agent_id, agent in self.agents.items():
                if agent_id in avg_grad and agent_id in self.meta_parameters:
                    with agent._q_lock:
                        self.meta_parameters[agent_id] += meta_lr * avg_grad[agent_id]
                        agent.q_table = self.meta_parameters[agent_id].clone()
                        if agent.config.use_double_q:
                            agent.target_q_table = self.meta_parameters[agent_id].clone()
            
            logger.debug(f"Updated meta-parameters with learning rate {meta_lr}")
        except Exception as e:
            logger.error(f"Error updating meta-parameters: {e}")
    
    async def meta_test(self, test_environments: List[Any], adaptation_episodes: int = None):
        """
        Test the meta-learned model on new tasks.
        """
        episodes = adaptation_episodes or self.config.agents_config[0].q_learning_config.adaptation_episodes
        results = []
        
        for i, env in enumerate(test_environments):
            # Reset to meta-parameters
            self._load_meta_parameters()
            
            # Quick adaptation
            await self.adapt_to_task(env, episodes)
            
            # Evaluate adapted policy
            total_reward = await self._evaluate_policy(env)
            results.append(total_reward)
            
            logger.info(f"Task {i+1} test result: {total_reward}")
        
        return results
    
    async def _evaluate_policy(self, environment, num_episodes: int = 10):
        """Evaluate the current policy on an environment"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = environment.reset()
            done = False
            episode_reward = 0
            steps = 0
            max_steps = self.config.agents_config[0].q_learning_config.max_steps
            
            while not done and steps < max_steps:
                try:
                    # Get action from responsible agent with no exploration
                    agent = self._get_responsible_agent(state)
                    with agent._param_lock:
                        original_epsilon = agent.epsilon
                        agent.epsilon = 0.0  # No exploration during evaluation
                    
                    action = agent.select_action(state)
                    
                    # Restore epsilon
                    with agent._param_lock:
                        agent.epsilon = original_epsilon
                    
                    # Take action
                    next_state, reward, done, _ = environment.step(action)
                    state = next_state
                    episode_reward += reward
                    steps += 1
                except Exception as e:
                    logger.error(f"Error during policy evaluation: {e}")
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes  # Return average reward

async def main():
    """
    Example of testing the hierarchical multi-agent system with MAML-style meta-learning.
    """
    # Define base grid world configuration
    base_grid_size = (20, 20)
    base_obstacles = [(5, 5), (5, 6), (5, 7), (5, 8)]
    
    # Create meta-learning tasks (different environment configurations)
    meta_tasks = [
        # Task 1: Original environment
        {"obstacles": base_obstacles},
        # Task 2: More obstacles
        {"obstacles": base_obstacles + [(15, 10), (15, 11), (15, 12)]},
        # Task 3: Different obstacle pattern
        {"obstacles": [(10, 10), (11, 10), (12, 10), (13, 10)]},
        # Task 4: Sparse obstacles
        {"obstacles": [(8, 8), (12, 12), (16, 16)]}
    ]
    
    # Define additional test tasks (unseen during meta-training)
    test_tasks = [
        # Test task 1: New obstacle pattern
        {"obstacles": [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]},
        # Test task 2: Maze-like pattern
        {"obstacles": [(i, 10) for i in range(15) if i != 7] + [(7, j) for j in range(10)]}
    ]
    
    # Create task environments
    task_environments = []
    for task in meta_tasks:
        env = HierarchicalMultiAgentEnvironment(base_grid_size, task["obstacles"])
        task_environments.append(env)
    
    test_environments = []
    for task in test_tasks:
        env = HierarchicalMultiAgentEnvironment(base_grid_size, task["obstacles"])
        test_environments.append(env)
    
    # Base agent configuration with MAML-specific parameters
    base_agent_config = AgentConfig(
        agent_id="meta_agent",
        q_learning_config=QLearningParams(
            state_space=base_grid_size,
            action_space=8,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.3,  # Lower epsilon for faster adaptation
            epsilon_min=0.01,
            epsilon_decay=0.995,
            exploration_strategy="epsilon_greedy",
            # MAML-specific parameters
            meta_lr=0.01,
            adapt_lr=0.1,
            meta_batch_size=len(meta_tasks),
            adaptation_steps=5,
            adaptation_episodes=20
        )
    )
    
    # System configuration
    system_config = HierarchicalAgentSystemConfig(
        agents_config=[base_agent_config],
        grid_size=base_grid_size,
        action_space=8,
        pathfinder_config=PathfinderConfig(
            grid_size=base_grid_size,
            allow_diagonal=True,
            concurrent_expansions=50  # Higher concurrency for better performance
        ),
        max_communication_threads=8,
        max_learning_threads=8,
        experience_sharing_threshold=0.5,
        coordination_frequency=10
    )
    
    # Log device information
    print(f"Using device: {DEVICE}")
    print("Starting MAML-style meta-learning for Q-Learning...")
    
    # Initialize the MAML-enhanced agent system
    maml_system = MAMLQAgentSystem(system_config)
    
    try:
        # Meta-training phase
        print("Meta-training phase:")
        meta_iterations = 100
        
        print(f"Training across {len(task_environments)} tasks for {meta_iterations} meta-iterations")
        await maml_system.meta_train(task_environments, meta_iterations)
        
        # Save meta-learned model
        os.makedirs("./models", exist_ok=True)
        maml_system.save_q_tables("./models/maml_meta_learned")
        print("Meta-learned model saved to ./models/maml_meta_learned")
        
        # Meta-testing phase
        print("\nMeta-testing phase:")
        print(f"Testing on {len(test_environments)} unseen tasks...")
        
        # Test with a fixed number of adaptation episodes
        adaptation_episodes = 20
        test_results = await maml_system.meta_test(test_environments, adaptation_episodes)
        
        for i, result in enumerate(test_results):
            print(f"Test task {i+1} average reward after {adaptation_episodes} adaptation episodes: {result:.2f}")
        
        # Quick visualization for the first test task
        print("\nVisualizing adapted policy on test task 1:")
        test_env = test_environments[0]
        test_env.reset()
        state = test_env.agent_pos
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 30:  # Show first 30 steps max
            action = await maml_system.select_action(state)
            next_state, reward, done, _ = test_env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            print(f"\nStep {steps}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(test_env.render())
            await asyncio.sleep(0.2)
        
        print(f"\nVisualization complete. Final reward: {total_reward:.2f}, Steps: {steps}")
        
    except Exception as e:
        print(f"Error during meta-learning: {e}")
    finally:
        # Clean up resources
        maml_system.close()
        print("\nMeta-learning process completed.")


if __name__ == "__main__":
    asyncio.run(main())
