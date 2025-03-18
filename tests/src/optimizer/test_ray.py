"""
Unit tests for the Ray distributed computing implementation.

This module tests the functionality of the RayDistributor class in both
Ray-available and Ray-unavailable scenarios.
"""

import unittest
import os
import sys
import asyncio
import warnings
import time
import logging
import tracemalloc
import gc
import contextlib
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List, Callable

import pytest
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock modules before importing real ones
# This ensures that even if Ray isn't installed, tests will run
mock_tune = MagicMock()
mock_tune.choice = MagicMock(return_value=True)
mock_tune.uniform = MagicMock(return_value=0.05)
mock_tune.loguniform = MagicMock(return_value=0.05)
mock_tune.randint = MagicMock(return_value=5)
mock_tune.lograndint = MagicMock(return_value=5)
mock_tune.run = MagicMock(return_value=MagicMock(best_config={"learning_rate": 0.05, "smart_action": True}))
mock_tune.report = MagicMock()

# Create a mock for the remote decorator that handles both ways it might be used
remote_mock = MagicMock()
remote_mock.remote.return_value = "task_id"

# Special handling for @ray.remote decorator pattern
def mock_remote_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        # Used as @ray.remote directly on a function
        decorated_func = args[0]
        wrapped = MagicMock(wraps=decorated_func)
        wrapped.remote = MagicMock(return_value="task_id")
        return wrapped
    else:
        # Used as @ray.remote(args, kwargs)
        return remote_mock

# Create a MagicMock that wraps the remote decorator function instead of using the function directly
mock_remote = MagicMock()
mock_remote.side_effect = mock_remote_decorator

mock_ray = MagicMock()
mock_ray.tune = mock_tune
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.init = MagicMock(return_value={"resources": {"CPU": 2.0}})
mock_ray.shutdown = MagicMock()
mock_ray.put = MagicMock(return_value="dataset_ref")
mock_ray.get = MagicMock(return_value=[{"params": {"hidden_size": 64}}])
mock_ray.cluster_resources = MagicMock(return_value={"CPU": 2.0})
mock_ray.remote = mock_remote  # Use the MagicMock wrapper instead of the function directly

# Mock schedulers
class MockScheduler:
    def __init__(self, *args, **kwargs):
        pass

# Mock Ray modules
sys.modules['ray'] = mock_ray
sys.modules['ray.tune'] = mock_tune
sys.modules['ray.tune.schedulers'] = MagicMock()
sys.modules['ray.tune.schedulers'].ASHAScheduler = MockScheduler
sys.modules['ray.tune.schedulers'].PopulationBasedTraining = MockScheduler
sys.modules['ray.tune.search'] = MagicMock()
sys.modules['ray.tune.search.optuna'] = MagicMock()
sys.modules['ray.tune.search.optuna'].OptunaSearch = MagicMock()

# Now import our code that depends on Ray
from src.optimizer.ray import RayDistributor, RayConfig, OptunaConfig

# Create a temp directory for Ray
RAY_TEMP_DIR = tempfile.mkdtemp(prefix="ray_test_")

# Configure environment to avoid Ray file operations
os.environ["RAY_DISABLE_MONITOR"] = "1"
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_DISABLE_BROWSER_OPENING"] = "1" 
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_DISABLE_LOG_MONITOR_WARNING"] = "1"
os.environ["RAY_TEMP_DIR"] = RAY_TEMP_DIR
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_LOG_TO_STDOUT"] = "0"

# Start tracemalloc to track ResourceWarning origins
tracemalloc.start()

# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)

# Filter ResourceWarnings to reduce noise
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=r"unclosed.*socket")
warnings.filterwarnings("ignore", message=r"unclosed.*file")
warnings.filterwarnings("ignore", message=r".*socket\.socketpair.*")
warnings.filterwarnings("ignore", message=r".*subprocess\.Popen.*")
warnings.filterwarnings("ignore", message=r".*EventLoop.*")
warnings.filterwarnings("ignore", message=r".*temp.*log.*")
warnings.filterwarnings("ignore", message=r".*There is no current event loop.*")
warnings.filterwarnings("ignore", message=r".*deprecated.*")

# Set HAS_RAY to True since we're using mock Ray
HAS_RAY = True
    
# Mock implementations for testing

class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self):
        self.state_shape = (4,)
        self.action_space = 2
        self.current_step = 0
        self.max_steps = 10
        
    def reset(self):
        self.current_step = 0
        return np.zeros(self.state_shape)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = 1.0 if action == 1 else 0.5
        next_state = np.ones(self.state_shape) * self.current_step / 10
        return next_state, reward, done, {}

class MockAgentSystem:
    """Mock agent system for testing."""
    
    def __init__(self, config):
        self.config = config
        self.action_value = 1 if getattr(config, "smart_action", False) else 0
        
    async def train(self, env, episodes):
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.action_value
                next_state, reward, done, _ = env.step(action)
                state = next_state
    
    async def select_action(self, state):
        return self.action_value
    
    def get_serializable_state(self):
        return {"config": self.config.__dict__ if hasattr(self.config, "__dict__") else self.config}
    
    def close(self):
        pass  # Added for compatibility with OptunaOptimizer

class MockMAMLAgent:
    """Mock MAML agent for testing."""
    
    def __init__(self, system_config, maml_config):
        self.system_config = system_config
        self.maml_config = maml_config
        
    async def meta_train(self, training_tasks, num_meta_iterations=10):
        # Simulate meta-training
        for _ in range(num_meta_iterations):
            for task_id, env in training_tasks.items():
                state = env.reset()
                done = False
                while not done:
                    action = 1 if getattr(self.system_config, "smart_action", False) else 0
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
    
    async def adapt_to_new_task(self, env, num_adaptation_steps=5):
        total_reward = 0
        for _ in range(num_adaptation_steps):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = 1 if getattr(self.system_config, "smart_action", False) else 0
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        return total_reward / num_adaptation_steps
    
    def get_serializable_state(self):
        return {
            "system_config": self.system_config.__dict__ if hasattr(self.system_config, "__dict__") else self.system_config,
            "maml_config": self.maml_config.__dict__ if hasattr(self.maml_config, "__dict__") else self.maml_config
        }

class MockNeuralNetwork:
    """Mock neural network for testing."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        
    def get_serializable_state(self):
        return {"params": self.params}
    
    def load_combined_state(self, states):
        # Just use the first state for testing purposes
        if states:
            self.params = states[0].get("params", {})
    
    def load_state(self, path):
        pass
    
    def save_state(self, path):
        pass

class MockConfig:
    """Mock configuration class."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Patch modules for testing
sys.modules["re_learning.q_learning"] = MagicMock()
sys.modules["re_learning.q_learning"].HierarchicalAgentSystem = MockAgentSystem
sys.modules["src.maml"] = MagicMock()
sys.modules["src.maml"].MAMLAgent = MockMAMLAgent

# Test cases
class TestRayDistributor(unittest.TestCase):
    """Test cases for the RayDistributor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources once for all tests."""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests are done."""
        # Clean up Ray temp directory
        try:
            import shutil
            if os.path.exists(RAY_TEMP_DIR):
                shutil.rmtree(RAY_TEMP_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up Ray temp directory: {e}")
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset mock calls before each test
        mock_ray.init.reset_mock()
        mock_ray.shutdown.reset_mock()
        mock_ray.is_initialized.reset_mock()
        mock_ray.remote.reset_mock()
        mock_ray.get.reset_mock()
        mock_ray.put.reset_mock()
        mock_tune.run.reset_mock()
        
        # Set default return values
        mock_ray.is_initialized.return_value = False
        
        # Custom patcher for static methods in Ray module
        # This is needed because the @ray.remote decorator is a staticmethod in ray.py
        self.remote_patcher = patch('src.optimizer.ray.ray.remote', mock_remote_decorator)
        self.remote_patcher.start()
        
        self.ray_config = RayConfig(
            address=None,
            num_cpus=2,
            num_gpus=0,
            memory=None,
            object_store_memory=None,
            redis_max_memory=None
        )
        
        self.optuna_config = OptunaConfig(
            study_name="test_study",
            direction="max",
            n_trials=5,
            timeout=60
        )
        
        self.env_creator = lambda: MockEnvironment()
        self.task_creators = [self.env_creator for _ in range(3)]
        self.test_task_creators = [self.env_creator for _ in range(2)]
        
        # Mock model factory and related functions
        self.model_factory = lambda **kwargs: MockNeuralNetwork(**kwargs)
        self.train_func = MagicMock()
        self.eval_func = MagicMock(return_value={"accuracy": 0.85})
        
        # For objective function
        self.objective_func = MagicMock(return_value=0.75)
        
        # Parameter variations for distributed testing
        self.base_config = MockConfig(learning_rate=0.01, smart_action=False)
        self.param_variations = [
            {"learning_rate": 0.01, "smart_action": False},
            {"learning_rate": 0.1, "smart_action": True}
        ]
        
        # For MAML testing
        self.base_maml_config = MockConfig(inner_lr=0.01)
        self.base_system_config = MockConfig(learning_rate=0.01, smart_action=False)
        self.maml_param_variations = [
            {"maml_inner_lr": 0.01, "learning_rate": 0.01},
            {"maml_inner_lr": 0.1, "smart_action": True}
        ]
        
        # For hyperparameter search - use compatible format with Ray Tune
        self.param_space = {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "smart_action": {"type": "categorical", "values": [True, False]}
        }
        
        # PBT-compatible format for hyperparam_mutations
        self.pbt_hyperparam_mutations = {
            "learning_rate": lambda: np.random.uniform(0.001, 0.1),
            "smart_action": [True, False]
        }
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop patch
        self.remote_patcher.stop()
        
        # Force garbage collection
        gc.collect()
    
    def test_initialize_ray(self):
        """Test Ray initialization."""
        # Configure mocks
        mock_ray.is_initialized.return_value = False
        
        distributor = RayDistributor(self.ray_config)
        
        # init should be called during initialization
        mock_ray.init.assert_called_once()
        self.assertTrue(distributor.is_initialized)
    
    def test_initialize_ray_already_running(self):
        """Test Ray initialization when Ray is already running."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        
        distributor = RayDistributor(self.ray_config)
        
        # init should not be called if Ray is already initialized
        mock_ray.init.assert_not_called()
        self.assertTrue(distributor.is_initialized)
    
    def test_distribute_q_learning_with_ray(self):
        """Test distribute_q_learning with Ray available."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        mock_ray.get.return_value = {"reward": 7.5, "agent_state": {"config": {}}}
        
        # Patch the _remote_train_agent method to use our mock
        with patch('src.optimizer.ray.RayDistributor._remote_train_agent') as mock_remote_train:
            mock_remote_train.remote = MagicMock(return_value="task_id")
            
            distributor = RayDistributor(self.ray_config)
            
            results = distributor.distribute_q_learning(
                self.env_creator,
                self.base_config,
                self.param_variations,
                episodes=10,
                eval_episodes=2
            )
            
            # Verify results
            self.assertEqual(len(results), 2)
            self.assertIn("reward", results[0])
            self.assertIn("params", results[0])
            
            # Verify the remote method was called
            mock_remote_train.remote.assert_called()
    
    def test_distribute_q_learning_without_ray(self):
        """Test distribute_q_learning without Ray."""
        distributor = RayDistributor(self.ray_config)
        distributor.is_initialized = False  # Force sequential mode
        
        results = distributor.distribute_q_learning(
            self.env_creator,
            self.base_config,
            self.param_variations,
            episodes=10,
            eval_episodes=2
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn("reward", results[0])
        self.assertIn("params", results[0])
        
        # The second config has smart_action=True which should get higher reward
        self.assertGreater(results[1]["reward"], results[0]["reward"])
    
    def test_distribute_maml_with_ray(self):
        """Test distribute_maml with Ray available."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        mock_ray.get.return_value = {"reward": 8.0, "agent_state": {"system_config": {}, "maml_config": {}}}
        
        # Patch the _remote_meta_train_agent method to use our mock
        with patch('src.optimizer.ray.RayDistributor._remote_meta_train_agent') as mock_remote_train:
            mock_remote_train.remote = MagicMock(return_value="task_id")
            
            distributor = RayDistributor(self.ray_config)
            
            results = distributor.distribute_maml(
                self.task_creators,
                self.test_task_creators,
                self.base_maml_config,
                self.base_system_config,
                self.maml_param_variations,
                num_meta_iterations=5,
                num_adaptation_steps=3
            )
            
            # Verify results
            self.assertEqual(len(results), 2)
            self.assertIn("reward", results[0])
            self.assertIn("params", results[0])
            
            # Verify the remote method was called
            mock_remote_train.remote.assert_called()
    
    def test_distribute_maml_without_ray(self):
        """Test distribute_maml without Ray."""
        distributor = RayDistributor(self.ray_config)
        distributor.is_initialized = False  # Force sequential mode
        
        results = distributor.distribute_maml(
            self.task_creators,
            self.test_task_creators,
            self.base_maml_config,
            self.base_system_config,
            self.maml_param_variations,
            num_meta_iterations=5,
            num_adaptation_steps=3
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn("reward", results[0])
        self.assertIn("params", results[0])
        
        # The second config has smart_action=True which should get higher reward
        self.assertGreater(results[1]["reward"], results[0]["reward"])
    
    def test_integrate_with_optuna(self):
        """Test integration with Optuna."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        mock_tune.run.return_value = MagicMock(best_config={"learning_rate": 0.05, "smart_action": True})
        
        distributor = RayDistributor(self.ray_config)
        
        result = distributor.integrate_with_optuna(
            self.optuna_config,
            self.objective_func,
            self.param_space,
            n_samples=2
        )
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn("learning_rate", result)
        self.assertIn("smart_action", result)
        
        # Verify Ray methods were called
        mock_tune.run.assert_called_once()
    
    def test_distribute_neural_network_training(self):
        """Test distribute_neural_network_training."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        mock_ray.put.return_value = "dataset_ref"
        mock_ray.get.return_value = [{"params": {"hidden_size": 64}}]
        
        # Patch the remote_train_and_eval method to use our mock
        with patch('src.optimizer.ray.ray.remote') as mock_remote:
            mock_fn = MagicMock()
            mock_fn.remote = MagicMock(return_value="task_id")
            mock_remote.return_value = mock_fn
            
            distributor = RayDistributor(self.ray_config)
            
            # Mock dataset
            mock_dataset = {"train": np.random.rand(100, 10), "labels": np.random.randint(0, 2, 100)}
            
            model, metrics = distributor.distribute_neural_network_training(
                self.model_factory,
                mock_dataset,
                self.train_func,
                self.eval_func,
                num_workers=2,
                batch_size=32,
                epochs=10
            )
            
            # Verify results
            self.assertIsInstance(model, MockNeuralNetwork)
            self.assertIsInstance(metrics, dict)
            
            # Verify Ray methods were called
            mock_ray.put.assert_called_once()
            mock_remote.assert_called()
    
    def test_pbt_optimize(self):
        """Test Population Based Training optimization."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        
        # Create a fresh mock for tune.run to avoid call count issues
        with patch('src.optimizer.ray.tune.run') as mock_tune_run:
            mock_tune_run.return_value = MagicMock(best_config={"learning_rate": 0.05, "smart_action": True})
            
            distributor = RayDistributor(self.ray_config)
            
            result = distributor.pbt_optimize(
                self.model_factory,
                self.env_creator,
                self.pbt_hyperparam_mutations,  # Use PBT-compatible format
                self.train_func,
                self.eval_func,
                population_size=4,
                num_iterations=5
            )
            
            # Verify results
            self.assertIsInstance(result, dict)
            self.assertIn("learning_rate", result)
            self.assertIn("smart_action", result)
            
            # Verify Ray methods were called
            mock_tune_run.assert_called_once()
    
    def test_shutdown(self):
        """Test Ray shutdown."""
        # Configure mocks
        mock_ray.is_initialized.return_value = True
        
        distributor = RayDistributor(self.ray_config)
        distributor.shutdown()
        
        # Verify Ray methods were called
        mock_ray.shutdown.assert_called_once()
        self.assertFalse(distributor.is_initialized)

# Custom test runner for better benchmarking
class BenchmarkTestResult(unittest.TextTestResult):
    """Custom TestResult that tracks and displays execution time for each test."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = {}
        
    def startTest(self, test):
        self.start_time = time.time()
        super().startTest(test)
        
    def addSuccess(self, test):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addSuccess(test)
            
    def _getShortTestName(self, test):
        """Extract the short test name without the test class and module."""
        return test._testMethodName
            
    def addError(self, test, err):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addError(test, err)
        
    def addFailure(self, test, err):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addFailure(test, err)
        
    def addSkip(self, test, reason):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addSkip(test, reason)
        
    def addExpectedFailure(self, test, err):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addExpectedFailure(test, err)
        
    def addUnexpectedSuccess(self, test):
        elapsed_time = time.time() - self.start_time
        test_name = self._getShortTestName(test)
        self.test_timings[test_name] = elapsed_time
        super().addUnexpectedSuccess(test)
    
    def printResults(self):
        self.stream.writeln("\n===== TEST BENCHMARK RESULTS =====")
        sorted_times = sorted(self.test_timings.items(), key=lambda x: x[1], reverse=True)
        
        max_name_length = max(len(name) for name, _ in sorted_times)
        
        # Get descriptions for all test names
        descriptions = {}
        for test in self.test_timings.keys():
            # Get the test method docstring for description
            func = getattr(TestRayDistributor, test)
            descriptions[test] = func.__doc__.strip() if func.__doc__ else test
        
        self.stream.writeln(f"{'Test Name'.ljust(max_name_length)} | {'Time (s)'.ljust(10)} | Description")
        self.stream.writeln('-' * (max_name_length + 10 + 50))
        
        for test_name, elapsed_time in sorted_times:
            padded_name = test_name.ljust(max_name_length)
            desc = descriptions.get(test_name, "")
            self.stream.writeln(f"{padded_name} | {elapsed_time:.6f}s | {desc}")
            
        self.stream.writeln('-' * (max_name_length + 10 + 50))
        self.stream.writeln(f"Total test time: {sum(self.test_timings.values()):.6f}s")
        self.stream.writeln("==================================")

class BenchmarkTestRunner(unittest.TextTestRunner):
    """Custom TestRunner that uses BenchmarkTestResult to track and display test times."""
    
    resultclass = BenchmarkTestResult
    
    def __init__(self, *args, **kwargs):
        self.quiet = kwargs.pop('quiet', False)
        super().__init__(*args, **kwargs)
        
    def run(self, test):
        # Temporarily redirect stdout to suppress output if quiet mode
        if self.quiet:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
        try:
            result = super().run(test)
            if hasattr(result, 'printResults'):
                # Restore stdout before printing results
                if self.quiet:
                    sys.stdout = original_stdout
                result.printResults()
            return result
        finally:
            # Ensure stdout is restored even if an exception occurs
            if self.quiet and sys.stdout != original_stdout:
                sys.stdout.close()
                sys.stdout = original_stdout

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRayDistributor)
    
    # Run the tests with our custom benchmark runner
    # Use quiet=True to only show benchmark results, not the test details
    runner = BenchmarkTestRunner(verbosity=1, quiet=False)
    runner.run(test_suite)
