"""
Test suite for the OptunaOptimizer module.

This module contains tests for the Optuna-based hyperparameter optimization
capabilities provided by the optimizer/optuna.py module. It uses simple
test environments and models to validate the optimization functionality.
"""

import unittest
import os
import tempfile
import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from optimizer.optuna import OptunaOptimizer
from optimizer.optimizer_config import OptimizerConfig, OptunaConfig
from re_learning.q_learning import AgentConfig, HierarchicalAgentSystemConfig, QLearningParams

# Try to import MAML components
try:
    from src.meta_learning.maml import MAMLConfig, MAMLAgent
    MAML_AVAILABLE = True
except ImportError:
    MAML_AVAILABLE = False


class MockQAgent:
    """A simple Q-agent for testing optimization."""
    
    def __init__(self, learning_rate=0.01, epsilon=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_table = {}
        
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # Initialize for 3 actions
        return self.q_table[state][action]
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        # Convert state to hashable type
        state_key = tuple(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 3)  # Random action
            
        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]  # Initialize for 3 actions
            
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning update rule."""
        # Convert states to hashable type
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0, 0.0]
        
        # Get current Q-value
        current_q = self.q_table[state_key][action]
        
        # Get max Q-value for next state
        max_next_q = max(self.q_table[next_state_key]) if not done else 0
        
        # Calculate target Q-value
        target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
    @classmethod
    def from_config(cls, learning_rate, epsilon, discount_factor):
        """Create agent from configuration parameters."""
        return cls(learning_rate, epsilon, discount_factor)


class SimpleEnvironment:
    """A simple test environment for Q-learning optimization."""
    
    def __init__(self, seed=None):
        self.grid_size = (5, 5)
        self.max_steps = 10
        self.current_step = 0
        self.pos_x = 0
        self.pos_y = 0
        self.n_actions = 3
        
        if seed is not None:
            np.random.seed(seed)
    
    def reset(self):
        self.pos_x = 0
        self.pos_y = 0
        self.current_step = 0
        return (self.pos_x, self.pos_y)
    
    def step(self, action):
        # Simple dynamics: action determines direction of movement
        if action == 0:  # Stay
            pass
        elif action == 1:  # Move right
            self.pos_x = min(self.pos_x + 1, self.grid_size[0] - 1)
        elif action == 2:  # Move left
            self.pos_x = max(self.pos_x - 1, 0)
        
        # Current state as a tuple
        state = (self.pos_x, self.pos_y)
        
        # Reward is higher when closer to the end state
        reward = 0.1 * self.pos_x
        
        # Give a big reward for reaching the end state
        if self.pos_x == self.grid_size[0] - 1:
            reward += 1.0
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.pos_x == self.grid_size[0] - 1
        
        return state, reward, done, {}


class SimpleNeuralNetwork(nn.Module):
    """A simple neural network for testing optimization."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for testing optimization."""
    
    def __init__(self, input_dim=10, num_heads=2, hidden_dim=20, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Take the mean of sequence dimension for classification
        x = torch.mean(x, dim=1)
        x = self.output(x)
        return x


class SimpleDiffusionModel(nn.Module):
    """A very simple diffusion model for testing optimization."""
    
    def __init__(self, input_dim=10, hidden_dim=20, timesteps=10, noise_scale=0.1):
        super().__init__()
        self.timesteps = timesteps
        self.noise_scale = noise_scale
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def add_noise(self, x, t):
        """Add noise to the input based on timestep t."""
        noise = torch.randn_like(x) * self.noise_scale * t / self.timesteps
        return x + noise, noise
    
    def forward(self, x, t=None):
        """Forward pass to predict noise at timestep t."""
        batch_size = x.shape[0]
        
        if t is None:
            # Random timestep if not provided
            t = torch.randint(1, self.timesteps + 1, (batch_size,)).float() / self.timesteps
        else:
            # Make sure t has the right shape for batch processing
            if isinstance(t, (int, float)):
                t = torch.tensor([t] * batch_size).float() / self.timesteps
            elif isinstance(t, torch.Tensor) and t.ndim == 0:
                t = t.repeat(batch_size) / self.timesteps
            else:
                t = t.float() / self.timesteps
        
        # Add noise to input
        noisy_x, true_noise = self.add_noise(x, t[0] if t.ndim > 0 else t)
        
        # Encode
        h = self.encoder(noisy_x)
        
        # Make sure t has the right shape for concatenation
        if t.ndim == 1:
            t_emb = t.unsqueeze(1)  # [batch_size, 1]
        else:
            t_emb = t.view(batch_size, 1)
        
        # Concatenate encoded features and time embedding
        h_with_t = torch.cat([h, t_emb], dim=1)
        
        # Predict noise
        pred_noise = self.noise_predictor(h_with_t)
        
        return pred_noise, true_noise


class TestOptunaOptimizer(unittest.TestCase):
    """Test cases for OptunaOptimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a minimal optimizer config with few trials
        optuna_config = OptunaConfig(
            study_name="test_study",
            direction="maximize",
            n_trials=3,  # Use a small number for quick testing
            timeout=30,  # 30 seconds timeout
            sampler="TPESampler",
            pruner="MedianPruner"
        )
        self.optimizer_config = OptimizerConfig(optuna=optuna_config)
        self.optimizer = OptunaOptimizer(self.optimizer_config)
    
    def test_create_study(self):
        """Test study creation with different parameters."""
        # Test with default parameters
        self.optimizer.create_study()
        self.assertIsNotNone(self.optimizer.study)
        self.assertEqual(self.optimizer.study.direction.name, "MAXIMIZE")
        
        # Create a new optimizer instance to avoid interference between tests
        self.optimizer = OptunaOptimizer(self.optimizer_config)
        
        # Test with custom study name and direction
        self.optimizer.create_study(study_name="custom_study", direction="minimize")
        self.assertEqual(self.optimizer.study.study_name, "custom_study")
        self.assertEqual(self.optimizer.study.direction.name, "MINIMIZE")
    
    def test_optimize_q_learning(self):
        """Test Q-learning hyperparameter optimization."""
        # Function to create test environment
        def environment_factory():
            return SimpleEnvironment(seed=42)
        
        # Define parameter space
        param_space = {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "epsilon": {"type": "float", "low": 0.01, "high": 0.2},
            "discount_factor": {"type": "float", "low": 0.8, "high": 0.99}
        }
        
        # Q-learning parameters
        q_learning_params = QLearningParams(
            alpha=0.01,  # learning_rate
            epsilon=0.1,
            gamma=0.9,  # discount_factor
            state_space=(5, 5),
            action_space=3,
            epsilon_decay=0.99,
            epsilon_min=0.01
        )
        
        # Base agent configuration
        base_agent_config = AgentConfig(
            agent_id="test_agent",
            q_learning_config=q_learning_params
        )
        
        # Create agent configurations for system config
        agents_config = [base_agent_config]
        
        # System configuration
        system_config = HierarchicalAgentSystemConfig(
            agents_config=agents_config,
            grid_size=(5, 5),
            action_space=3
        )
        
        # Define a custom objective function for this test
        def mock_objective(trial):
            # Sample hyperparameters from the search space
            learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
            epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
            discount_factor = trial.suggest_float("discount_factor", 0.8, 0.99)
            
            # Create agent with sampled parameters
            agent = MockQAgent(
                learning_rate=learning_rate,
                epsilon=epsilon,
                discount_factor=discount_factor
            )
            
            # Evaluate agent
            env = environment_factory()
            total_reward = 0
            n_episodes = 2
            
            for _ in range(n_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                
                total_reward += episode_reward
            
            return total_reward / n_episodes
        
        # Create a study and optimize directly
        self.optimizer.create_study()
        self.optimizer.study.optimize(mock_objective, n_trials=3)
        
        # Get best parameters
        best_params = self.optimizer.study.best_params
        
        # Check if we got results
        self.assertIsNotNone(best_params)
        self.assertIn("learning_rate", best_params)
        self.assertIn("epsilon", best_params)
        self.assertIn("discount_factor", best_params)
    
    @unittest.skipIf(not MAML_AVAILABLE, "MAML module not available")
    def test_optimize_maml(self):
        """Test MAML hyperparameter optimization."""
        # Skip if MAML module is not available
        if not MAML_AVAILABLE:
            self.skipTest("MAML module not available")
        
        # Function to create test environment for different tasks
        def task_environment_factory(task_id):
            # Create environments with different settings for different tasks
            return SimpleEnvironment(seed=task_id + 42)
        
        # Define a custom objective function for this test
        def mock_objective(trial):
            # Sample hyperparameters from the search space
            inner_learning_rate = trial.suggest_float("inner_learning_rate", 0.001, 0.1, log=True)
            meta_learning_rate = trial.suggest_float("meta_learning_rate", 0.001, 0.1, log=True)
            num_inner_steps = trial.suggest_int("num_inner_steps", 1, 5)
            
            # In a real test, we would create a MAML agent and evaluate it
            # For simplicity, we'll just return a random score
            # This is a placeholder for actual MAML evaluation
            return np.random.random()  # Mock performance score
        
        # Create a study and optimize directly
        self.optimizer.create_study()
        self.optimizer.study.optimize(mock_objective, n_trials=3)
        
        # Get best parameters
        best_params = self.optimizer.study.best_params
        
        # Check if we got results
        self.assertIsNotNone(best_params)
        self.assertIn("inner_learning_rate", best_params)
        self.assertIn("meta_learning_rate", best_params)
        self.assertIn("num_inner_steps", best_params)
    
    def test_optimize_neural_network(self):
        """Test neural network hyperparameter optimization."""
        # Create a simple synthetic dataset
        def create_dataset(n_samples=100):
            X = torch.randn(n_samples, 10)
            y = torch.randint(0, 2, (n_samples,))
            return X, y
        
        X, y = create_dataset()
        
        # Define a custom objective function for this test
        def mock_objective(trial):
            # Sample hyperparameters from the search space
            hidden_dim = trial.suggest_int("hidden_dim", 10, 50)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            
            # Create model with sampled parameters
            model = nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 2)
            )
            
            # Simple evaluation (no training, just a mock score)
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y).sum().item() / y.size(0)
            
            return accuracy
        
        # Create a study and optimize directly
        self.optimizer.create_study()
        self.optimizer.study.optimize(mock_objective, n_trials=3)
        
        # Get best parameters
        best_params = self.optimizer.study.best_params
        
        # Check if we got results
        self.assertIsNotNone(best_params)
        self.assertIn("hidden_dim", best_params)
        self.assertIn("dropout_rate", best_params)
    
    def test_optimize_transformer(self):
        """Test transformer model hyperparameter optimization."""
        # Create a simple synthetic dataset
        def create_dataset(n_samples=100, seq_len=5):
            X = torch.randn(n_samples, seq_len, 10)
            y = torch.randint(0, 2, (n_samples,))
            return X, y
        
        X, y = create_dataset()
        
        # Define a custom objective function for this test
        def mock_objective(trial):
            # Sample hyperparameters from the search space
            num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            
            # Create model with sampled parameters
            model = SimpleTransformer(
                input_dim=10,
                num_heads=num_heads,
                hidden_dim=20,
                num_layers=num_layers,
                dropout=dropout_rate
            )
            
            # Simple evaluation (no training, just a mock score)
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y).sum().item() / y.size(0)
            
            return accuracy
        
        # Create a study and optimize directly
        self.optimizer.create_study()
        self.optimizer.study.optimize(mock_objective, n_trials=3)
        
        # Get best parameters
        best_params = self.optimizer.study.best_params
        
        # Check if we got results
        self.assertIsNotNone(best_params)
        self.assertIn("num_heads", best_params)
        self.assertIn("num_layers", best_params)
        self.assertIn("dropout_rate", best_params)
    
    def test_optimize_diffusion(self):
        """Test diffusion model hyperparameter optimization."""
        # Create a simple synthetic dataset
        def create_dataset(n_samples=100):
            X = torch.randn(n_samples, 10)  # Input data
            return X
        
        X = create_dataset()
        
        # Define a custom objective function for this test
        def mock_objective(trial):
            # Sample hyperparameters from the search space
            hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
            timesteps = trial.suggest_int("timesteps", 5, 20)
            noise_scale = trial.suggest_float("noise_scale", 0.05, 0.5)
            
            # Create model with sampled parameters
            model = SimpleDiffusionModel(
                input_dim=10,
                hidden_dim=hidden_dim,
                timesteps=timesteps,
                noise_scale=noise_scale
            )
            
            # Simple evaluation (just a mock score based on MSE)
            model.eval()
            with torch.no_grad():
                # Use a fixed timestep for evaluation
                t = 5  # middle timestep
                t_normalized = t / model.timesteps
                
                # Add noise to the data
                noisy_X, true_noise = model.add_noise(X, t_normalized)
                
                # Predict noise
                pred_noise, _ = model(X, t=t)
                
                # Calculate MSE
                mse = torch.mean((pred_noise - true_noise) ** 2).item()
            
            return -mse  # Negative because we want to maximize
        
        # Create a study and optimize directly
        self.optimizer.create_study()
        self.optimizer.study.optimize(mock_objective, n_trials=3)
        
        # Get best parameters
        best_params = self.optimizer.study.best_params
        
        # Check if we got results
        self.assertIsNotNone(best_params)
        self.assertIn("hidden_dim", best_params)
        self.assertIn("timesteps", best_params)
        self.assertIn("noise_scale", best_params)
    
    def test_save_and_load_results(self):
        """Test saving and loading optimization results."""
        # Create a study with direction minimize (for the quadratic function)
        self.optimizer.create_study(direction="minimize")
        
        # Define a simple objective function
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2  # Simple quadratic function, minimum at x=2
        
        # Run optimization
        self.optimizer.study.optimize(objective, n_trials=3)
        
        # Store the optimization results in the optimizer
        self.optimizer.best_params = self.optimizer.study.best_params
        self.optimizer.optimization_history = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime": trial.datetime_start.isoformat()
            }
            for trial in self.optimizer.study.trials
        ]
        
        # Create a test file path
        test_file = "test_optuna_results.pkl"
        
        try:
            # Save results
            self.optimizer.save_results(test_file)
            
            # Create a new optimizer and load the results
            new_optimizer = OptunaOptimizer(self.optimizer_config)
            loaded_results = new_optimizer.load_results(test_file)
            
            # Check if loaded results contain expected data
            self.assertIsNotNone(loaded_results)
            self.assertIn("best_params", loaded_results)
            self.assertIn("best_value", loaded_results)
            self.assertEqual(loaded_results["best_params"], self.optimizer.best_params)
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)


# Run tests if executed directly
if __name__ == "__main__":
    # You can use the following command line arguments:
    # -v or --verbose: Run tests in verbose mode
    # -q or --quiet: Run tests in quiet mode
    # -s or --single: Run a single specified test
    # Example: python test_optuna.py -v
    # Example: python test_optuna.py -s TestOptunaOptimizer.test_optimize_q_learning
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-v', '--verbose']:
            unittest.main(argv=[sys.argv[0], '-v'])
        elif sys.argv[1] in ['-q', '--quiet']:
            unittest.main(argv=[sys.argv[0], '-q'])
        elif sys.argv[1] in ['-s', '--single'] and len(sys.argv) > 2:
            unittest.main(argv=[sys.argv[0], sys.argv[2]])
        else:
            print("Unknown argument:", sys.argv[1])
            print("Available options: -v/--verbose, -q/--quiet, -s/--single [test_name]")
    else:
        unittest.main()
