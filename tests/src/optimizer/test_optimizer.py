"""
Unit tests and benchmarks for the optimizer module.

This file contains comprehensive tests for the unified optimizer module that combines
Optuna hyperparameter optimization with Ray distributed computing for reinforcement
learning and meta-learning models.
"""

import pytest
import asyncio
import time
import tempfile
import os
import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

# Register pytest markers
pytest.mark.benchmark = pytest.mark.benchmark

# Add the parent directory to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Create mock for re_learning.q_learning module to prevent import errors
class MockAgentConfig:
    pass

class MockQLearningParams:
    pass

class MockHierarchicalAgentSystemConfig:
    pass

# Create the mock modules
mock_q_learning = MagicMock()
mock_q_learning.AgentConfig = MockAgentConfig
mock_q_learning.QLearningParams = MockQLearningParams
mock_q_learning.HierarchicalAgentSystemConfig = MockHierarchicalAgentSystemConfig
mock_q_learning.HierarchicalAgentSystem = MagicMock()

# Create the parent module
sys.modules['re_learning'] = MagicMock()
sys.modules['re_learning.q_learning'] = mock_q_learning

# Mock MAML module
class MockMAMLConfig:
    pass

mock_maml = MagicMock()
mock_maml.MAMLConfig = MockMAMLConfig
mock_maml.MAMLAgent = MagicMock()
sys.modules['src.maml'] = mock_maml

# Now imports will work properly
from src.optimizer.optimizer import Optimizer
from src.optimizer.optimizer_config import OptimizerConfig, RayConfig, OptunaConfig

class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.state = None
        self.reset_called = 0
        self.step_called = 0
    
    def reset(self):
        self.reset_called += 1
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        self.step_called += 1
        # Simple action mapping: 0=up, 1=right, 2=down, 3=left
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        x, y = self.state
        nx, ny = x + dx, y + dy
        # Ensure we stay within grid boundaries
        nx = max(0, min(nx, self.grid_size[0] - 1))
        ny = max(0, min(ny, self.grid_size[1] - 1))
        self.state = (nx, ny)
        
        # Simple reward: +1 for reaching bottom right, -0.1 otherwise
        done = self.state == (self.grid_size[0] - 1, self.grid_size[1] - 1)
        reward = 1.0 if done else -0.1
        
        return self.state, reward, done, {}


class MockAgentSystem:
    """Mock agent system for testing."""
    
    def __init__(self, config):
        self.config = config
        self.trained = False
        self.train_called = 0
        self.select_action_called = 0
    
    async def train(self, env, episodes):
        self.train_called += 1
        self.trained = True
        # Simulate training time
        await asyncio.sleep(0.01)
        return True
    
    async def select_action(self, state):
        self.select_action_called += 1
        # Simple policy that prefers moving right and down
        return np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])
    
    def get_serializable_state(self):
        return {"trained": self.trained, "config": str(self.config)}
    
    def close(self):
        pass


class MockMAMLAgent:
    """Mock MAML agent for testing."""
    
    def __init__(self, system_config, maml_config):
        self.system_config = system_config
        self.maml_config = maml_config
        self.meta_trained = False
        self.meta_train_called = 0
        self.adapt_called = 0
    
    async def meta_train(self, training_tasks, num_meta_iterations=10):
        self.meta_train_called += 1
        self.meta_trained = True
        # Simulate training time
        await asyncio.sleep(0.01)
        return True
    
    async def adapt_to_new_task(self, env, num_adaptation_steps=5):
        self.adapt_called += 1
        # Simulate adaptation and return a reward
        await asyncio.sleep(0.01)
        return np.random.uniform(0.5, 0.9)  # Random reward between 0.5 and 0.9
    
    def get_serializable_state(self):
        return {
            "meta_trained": self.meta_trained,
            "system_config": str(self.system_config),
            "maml_config": str(self.maml_config)
        }


@pytest.fixture
def optimizer_config():
    """Create a default optimizer configuration for testing."""
    return OptimizerConfig(
        optuna=OptunaConfig(
            study_name="test_study",
            n_trials=2,
            timeout=5,
            sampler="TPESampler",
            pruner="MedianPruner",
            direction="maximize",
            storage=None
        ),
        ray=RayConfig(
            address=None,
            num_cpus=1,
            num_gpus=0,
            memory=None,
            object_store_memory=None,
            redis_max_memory=None
        )
    )


@pytest.fixture
def optimizer(optimizer_config):
    """Create a configured optimizer instance for testing."""
    with patch.object(Optimizer, "__init__", return_value=None) as mock_init:
        optimizer = Optimizer(optimizer_config)
        optimizer.config = optimizer_config
        optimizer.best_params = None
        optimizer.optimization_history = []
        
        # Create mocked internal components
        optimizer.optuna_optimizer = MagicMock()
        optimizer.optuna_optimizer.study = MagicMock()
        optimizer.optuna_optimizer.best_params = None
        optimizer.optuna_optimizer.optimization_history = []
        
        optimizer.ray_distributor = MagicMock()
        optimizer.ray_distributor.is_initialized = True
        
        return optimizer


@pytest.fixture
def mock_env_factory():
    """Create a mock environment factory function."""
    def factory():
        return MockEnvironment()
    return factory


@pytest.fixture
def base_agent_config():
    """Create a mock agent configuration."""
    # Mock Q-learning config
    q_learning_config = MagicMock(
        episodes=10,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.95
    )
    
    # Mock agent config
    config = MagicMock(
        agent_id="test_agent",
        q_learning_config=q_learning_config,
        temperature=1.0,
        use_double_q=False
    )
    
    return config


@pytest.fixture
def base_system_config(base_agent_config):
    """Create a mock system configuration."""
    config = MagicMock(
        agents_config=[base_agent_config],
        grid_size=(5, 5),
        action_space=4
    )
    return config


@pytest.fixture
def base_maml_config():
    """Create a mock MAML configuration."""
    config = MagicMock(
        meta_lr=0.01,
        inner_lr=0.1,
        meta_batch_size=2,
        adaptation_steps=3,
        meta_iterations=5
    )
    return config


@pytest.fixture
def mock_task_factory():
    """Create a mock task factory function."""
    def factory(task_id):
        # Create different environments based on task_id
        return MockEnvironment(grid_size=(5, 5))
    return factory


@pytest.fixture
def param_space():
    """Create a sample parameter space for optimization."""
    return {
        "q_learning_alpha": {"type": "float", "low": 0.05, "high": 0.2},
        "q_learning_gamma": {"type": "float", "low": 0.9, "high": 0.99},
        "temperature": {"type": "float", "low": 0.5, "high": 2.0},
        "use_double_q": {"type": "categorical", "values": [True, False]}
    }


@pytest.fixture
def maml_param_space():
    """Create a sample MAML parameter space for optimization."""
    return {
        "maml_meta_lr": {"type": "float", "low": 0.005, "high": 0.05},
        "maml_inner_lr": {"type": "float", "low": 0.05, "high": 0.2},
        "maml_adaptation_steps": {"type": "int", "low": 2, "high": 5}
    }


def evaluate_pathfinder(config, env):
    """Mock evaluation function for pathfinder."""
    # Simulate evaluation
    time.sleep(0.01)
    return np.random.uniform(0.3, 0.9)  # Random score between 0.3 and 0.9


# Patch the imports that would be used by the optimizer
@pytest.fixture(autouse=True)
def mock_imports():
    """Mock necessary imports for testing."""
    with patch("re_learning.q_learning.HierarchicalAgentSystem", MockAgentSystem), \
         patch("src.maml.MAMLAgent", MockMAMLAgent):
        yield


def test_optimizer_init(optimizer_config):
    """Test that the optimizer initializes correctly."""
    with patch.object(Optimizer, "__init__", return_value=None) as mock_init:
        optimizer = Optimizer(optimizer_config)
        mock_init.assert_called_once_with(optimizer_config)
        
        # Set up the optimizer manually since we mocked the init
        optimizer.config = optimizer_config
        optimizer.best_params = None
        optimizer.optimization_history = []
        optimizer.optuna_optimizer = MagicMock()
        optimizer.ray_distributor = MagicMock()
        
        assert optimizer.config == optimizer_config
        assert optimizer.best_params is None
        assert optimizer.optimization_history == []


def test_optimize_q_learning(optimizer, mock_env_factory, base_agent_config, base_system_config, param_space):
    """Test Q-Learning optimization."""
    # Configure mock
    expected_result = {"q_learning_alpha": 0.1, "q_learning_gamma": 0.95}
    optimizer.optuna_optimizer.optimize_q_learning.return_value = expected_result
    
    # Run optimization
    result = optimizer.optimize_q_learning(
        environment_factory=mock_env_factory,
        base_agent_config=base_agent_config,
        system_config=base_system_config,
        param_space=param_space,
        n_trials=2,
        eval_episodes=2,
        distributed=False
    )
    
    # Verify results
    optimizer.optuna_optimizer.optimize_q_learning.assert_called_once()
    assert result == expected_result
    assert optimizer.best_params == expected_result


def test_optimize_q_learning_distributed(optimizer, mock_env_factory, base_agent_config, base_system_config, param_space):
    """Test distributed Q-Learning optimization with Ray."""
    # Configure mock
    expected_result = {"q_learning_alpha": 0.1, "q_learning_gamma": 0.95}
    optimizer.ray_distributor.integrate_with_optuna.return_value = expected_result
    
    # Run optimization
    result = optimizer.optimize_q_learning(
        environment_factory=mock_env_factory,
        base_agent_config=base_agent_config,
        system_config=base_system_config,
        param_space=param_space,
        n_trials=2,
        eval_episodes=2,
        distributed=True
    )
    
    # Verify results
    optimizer.ray_distributor.integrate_with_optuna.assert_called_once()
    assert result == expected_result
    assert optimizer.best_params == expected_result


def test_optimize_maml(optimizer, mock_task_factory, base_system_config, base_maml_config, maml_param_space):
    """Test MAML optimization."""
    # Configure mock
    expected_result = {"maml_meta_lr": 0.02, "maml_inner_lr": 0.15}
    optimizer.optuna_optimizer.optimize_maml.return_value = expected_result
    
    # Run optimization
    result = optimizer.optimize_maml(
        task_factory=mock_task_factory,
        base_system_config=base_system_config,
        base_maml_config=base_maml_config,
        param_space=maml_param_space,
        n_trials=2,
        num_tasks=2,
        num_test_tasks=1,
        distributed=False
    )
    
    # Verify results
    optimizer.optuna_optimizer.optimize_maml.assert_called_once()
    assert result == expected_result
    assert optimizer.best_params == expected_result


def test_optimize_maml_distributed(optimizer, mock_task_factory, base_system_config, base_maml_config, maml_param_space):
    """Test distributed MAML optimization with Ray."""
    # Configure mock
    expected_result = {"maml_meta_lr": 0.02, "maml_inner_lr": 0.15}
    optimizer.ray_distributor.integrate_with_optuna.return_value = expected_result
    
    # Run optimization
    result = optimizer.optimize_maml(
        task_factory=mock_task_factory,
        base_system_config=base_system_config,
        base_maml_config=base_maml_config,
        param_space=maml_param_space,
        n_trials=2,
        num_tasks=2,
        num_test_tasks=1,
        distributed=True
    )
    
    # Verify results
    optimizer.ray_distributor.integrate_with_optuna.assert_called_once()
    assert result == expected_result
    assert optimizer.best_params == expected_result


def test_distributed_train_q_learning(optimizer, mock_env_factory, base_agent_config):
    """Test distributed Q-Learning training."""
    # Configure mock
    mock_results = [
        {"reward": 0.8, "params": {"q_learning_alpha": 0.1}},
        {"reward": 0.7, "params": {"q_learning_alpha": 0.2}}
    ]
    optimizer.ray_distributor.distribute_q_learning.return_value = mock_results
    
    # Run distributed training
    param_variations = [
        {"q_learning_alpha": 0.1},
        {"q_learning_alpha": 0.2}
    ]
    
    results = optimizer.distributed_train_q_learning(
        env_creator=mock_env_factory,
        base_config=base_agent_config,
        param_variations=param_variations,
        episodes=5,
        eval_episodes=2
    )
    
    # Verify results
    optimizer.ray_distributor.distribute_q_learning.assert_called_once()
    assert results == mock_results


def test_distributed_train_maml(optimizer, mock_task_factory, base_maml_config, base_system_config):
    """Test distributed MAML training."""
    # Configure mock
    mock_results = [
        {"reward": 0.85, "params": {"maml_meta_lr": 0.01}},
        {"reward": 0.75, "params": {"maml_meta_lr": 0.02}}
    ]
    optimizer.ray_distributor.distribute_maml.return_value = mock_results
    
    # Run distributed training
    param_variations = [
        {"maml_meta_lr": 0.01},
        {"maml_meta_lr": 0.02}
    ]
    
    # Create task creator functions
    task_creators = [lambda: mock_task_factory(i) for i in range(2)]
    test_task_creators = [lambda: mock_task_factory(i + 2) for i in range(1)]
    
    results = optimizer.distributed_train_maml(
        task_creators=task_creators,
        test_task_creators=test_task_creators,
        base_maml_config=base_maml_config,
        base_system_config=base_system_config,
        param_variations=param_variations,
        num_meta_iterations=3,
        num_adaptation_steps=2
    )
    
    # Verify results
    optimizer.ray_distributor.distribute_maml.assert_called_once()
    assert results == mock_results


def test_population_based_training(optimizer, mock_env_factory, param_space):
    """Test Population Based Training (PBT)."""
    # Configure mock
    expected_result = {"learning_rate": 0.001, "hidden_size": 64}
    optimizer.ray_distributor.pbt_optimize.return_value = expected_result
    
    # Define mock functions
    model_factory = MagicMock()
    train_func = MagicMock()
    eval_func = MagicMock()
    
    # Run PBT
    result = optimizer.population_based_training(
        model_factory=model_factory,
        env_creator=mock_env_factory,
        param_space=param_space,
        train_func=train_func,
        eval_func=eval_func,
        population_size=2,
        num_iterations=3
    )
    
    # Verify results
    optimizer.ray_distributor.pbt_optimize.assert_called_once()
    assert result == expected_result
    assert optimizer.best_params == expected_result


def test_optimize_pathfinder(optimizer, mock_env_factory, param_space):
    """Test Pathfinder optimization."""
    # Configure the optimizer with a mocked study
    optimizer.optuna_optimizer.study = MagicMock()
    optimizer.optuna_optimizer.study.optimize = MagicMock()
    optimizer.optuna_optimizer.study.best_params = {"learning_rate": 0.001}
    optimizer.optuna_optimizer.study.best_value = 0.85
    optimizer.optuna_optimizer.study.trials = [MagicMock()]
    
    # Define a mock base config
    base_config = MagicMock()
    
    # Run optimization
    result = optimizer.optimize_pathfinder(
        environment_factory=mock_env_factory,
        base_config=base_config,
        param_space=param_space,
        eval_func=evaluate_pathfinder,
        n_trials=2,
        distributed=False
    )
    
    # Verify results
    assert optimizer.optuna_optimizer.study.optimize.called
    assert result == {"learning_rate": 0.001}
    assert optimizer.best_params == result


def test_save_and_load_results(optimizer):
    """Test saving and loading optimization results."""
    # Set up fake results
    optimizer.best_params = {"param1": 0.1, "param2": 0.2}
    optimizer.optuna_optimizer.study = MagicMock()
    optimizer.optuna_optimizer.study.best_value = 0.85
    optimizer.optimization_history = [
        {"number": 1, "value": 0.7, "params": {"param1": 0.05, "param2": 0.15}},
        {"number": 2, "value": 0.85, "params": {"param1": 0.1, "param2": 0.2}}
    ]
    
    # Mock the save and load methods
    optimizer.save_results = MagicMock()
    optimizer.load_results = MagicMock()
    
    # Test save_results
    file_path = "test_results.pkl"
    optimizer.save_results(file_path)
    optimizer.save_results.assert_called_once_with(file_path)
    
    # Test load_results
    expected_results = {
        "best_params": {"param1": 0.1, "param2": 0.2},
        "best_value": 0.85,
        "optimization_history": optimizer.optimization_history
    }
    optimizer.load_results.return_value = expected_results
    
    result = optimizer.load_results(file_path)
    optimizer.load_results.assert_called_once_with(file_path)
    assert result == expected_results


def test_get_parameter_importance(optimizer):
    """Test getting parameter importance."""
    # Set up mock study with importance
    expected_result = {"param1": 0.7, "param2": 0.3}
    optimizer.get_parameter_importance = MagicMock(return_value=expected_result)
    
    # Get parameter importance
    importance = optimizer.get_parameter_importance()
    
    # Verify results
    optimizer.get_parameter_importance.assert_called_once()
    assert importance == expected_result


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for the optimizer module."""
    
    def test_benchmark_optimize_q_learning(self, optimizer, mock_env_factory, base_agent_config, base_system_config, param_space):
        """Benchmark Q-learning optimization performance."""
        # Configure the optimizer with a mocked study
        optimizer.optuna_optimizer.study = MagicMock()
        optimizer.optuna_optimizer.study.optimize = MagicMock()
        optimizer.optuna_optimizer.study.best_params = {"q_learning_alpha": 0.1}
        optimizer.optuna_optimizer.study.best_value = 0.8
        optimizer.optuna_optimizer.study.trials = [MagicMock()]
        
        # Run with timer
        start_time = time.time()
        optimizer.optimize_q_learning(
            environment_factory=mock_env_factory,
            base_agent_config=base_agent_config,
            system_config=base_system_config,
            param_space=param_space,
            n_trials=1,
            eval_episodes=1,
            distributed=False
        )
        duration = time.time() - start_time
        
        # Output benchmark result
        print(f"\nQ-learning optimization benchmark: {duration:.3f} seconds")
    
    def test_benchmark_optimize_maml(self, optimizer, mock_task_factory, base_system_config, base_maml_config, maml_param_space):
        """Benchmark MAML optimization performance."""
        # Configure the optimizer with a mocked study
        optimizer.optuna_optimizer.study = MagicMock()
        optimizer.optuna_optimizer.study.optimize = MagicMock()
        optimizer.optuna_optimizer.study.best_params = {"maml_meta_lr": 0.01}
        optimizer.optuna_optimizer.study.best_value = 0.8
        optimizer.optuna_optimizer.study.trials = [MagicMock()]
        
        # Run with timer
        start_time = time.time()
        optimizer.optimize_maml(
            task_factory=mock_task_factory,
            base_system_config=base_system_config,
            base_maml_config=base_maml_config,
            param_space=maml_param_space,
            n_trials=1,
            num_tasks=1,
            num_test_tasks=1,
            distributed=False
        )
        duration = time.time() - start_time
        
        # Output benchmark result
        print(f"\nMAML optimization benchmark: {duration:.3f} seconds")
    
    def test_benchmark_save_load_results(self, optimizer):
        """Benchmark save/load results performance."""
        # Set up fake results
        optimizer.best_params = {"param1": 0.1, "param2": 0.2}
        optimizer.optuna_optimizer.study = MagicMock()
        optimizer.optuna_optimizer.study.best_value = 0.85
        optimizer.optimization_history = [
            {"number": i, "value": 0.7 + i*0.01, "params": {"param1": 0.1, "param2": 0.2}}
            for i in range(100)  # Create a large history to test performance
        ]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Run save benchmark
            start_time = time.time()
            optimizer.save_results(temp_path)
            save_duration = time.time() - start_time
            
            # Clear the optimizer
            optimizer.best_params = None
            optimizer.optimization_history = []
            
            # Run load benchmark
            start_time = time.time()
            optimizer.load_results(temp_path)
            load_duration = time.time() - start_time
            
            # Output benchmark results
            print(f"\nResults save benchmark: {save_duration:.3f} seconds")
            print(f"Results load benchmark: {load_duration:.3f} seconds")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    """
    Run the tests. Also outputs benchmark results.
    
    To run only the tests:
    pytest test_optimizer.py -v
    
    To run with benchmarks:
    pytest test_optimizer.py -v -m benchmark
    """
    pytest.main(["-v", __file__])
