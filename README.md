# MetaQ-Star Machine Learning Framework

## A Novel Machine Learning Framework Combining Meta-Learning and Q-Learning with Pathfinding Optimization

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-orange.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/cuda-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

MetaQ-Star is a pioneering machine learning framework that combines the power of meta-learning with reinforcement learning, specifically Q-learning, enhanced by optimized pathfinding algorithms. This unique integration enables models to adapt quickly to new tasks with minimal data while efficiently exploring solution spaces.

### Core Innovation

MetaQ-Star introduces a novel approach where meta-learning principles enable rapid adaptation across tasks, while Q-learning provides robust reinforcement learning capabilities. The framework's pathfinding component optimizes the exploration of complex solution spaces, leading to faster convergence and better generalization.

## Key Components

- **Meta-Learning Engine**: Implements newly developed Mode-Conditional Bayesian Model-Agnostic Meta-Learning framework for fast adaptation to new tasks
- **Q-Learning System**: Advanced reinforcement learning implementation using a factorized, hierarchical double-agent Q-Learning algorithm
- **Pathfinder**: A* inspired optimization using a bi-directional and diagonal search algorithm for efficient solution space exploration in Q-tables
- **Hyperparameter Optimization**: Hyperparameter tuning and optimization using Optuna
- **Distributed Computation**: Ray integration for scaling across computational resources
- **Cache Management**: Intelligent caching system for optimized memory usage
- **Comprehensive Logging**: Detailed tracking of experiments and model performance
- **Database Integration**: Configurable database solutions for different environments

## Technical Architecture

MetaQ-Star is built on a modern Python stack:

- Python 3.12
- PyTorch 2.6
- CUDA 12.6
- Pydantic V2
- Optuna v4.2.1
- Ray 2.43.0
- Multi-device support: CUDA (preferred), MPS (secondary), CPU (backup)
- Cloud support for using MongoDB and Redis
- Local support for using SQLite3 and system memory

## Getting Started

### Currently under development: v0.25

## Documentation

For detailed documentation, please refer to the [research_docs](./research_docs) directory which includes implementation details and theoretical background.

## License

[MIT License](LICENSE)

## Citation

If you use MetaQ-Star in your research, please cite:

```text
@software{metaq_star2023,
  author = saviornt,
  title = MetaQ-Star: A Novel Machine Learning Framework,
  year = 2025,
  url = {https://github.com/saviornt/MetaQ-Star}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and join the development effort.
