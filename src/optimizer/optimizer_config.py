# ./src/optimizer/optimizer_config.py

from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional


class OptunaConfig(BaseModel):
    """
    Configuration settings for Optuna hyperparameter optimization.
    """
    study_name: str = Field(default="hyperparameter_study", description="Name of the Optuna study.")
    direction: str = Field(default="maximize", description="Optimization direction: 'minimize' or 'maximize'.")
    n_trials: int = Field(default=50, description="Number of hyperparameter trials to execute.")
    timeout: Optional[int] = Field(default=None, description="Maximum time (in seconds) for the optimization.")
    storage: Optional[str] = Field(default=None, description="Database URL for storing study results.")
    sampler: Optional[str] = Field(default="TPESampler", description="Sampler to use for the study.")
    pruner: Optional[str] = Field(default="MedianPruner", description="Pruner to use for the study.")


class RayConfig(BaseModel):
    """
    Configuration settings for Ray parallel execution.
    """
    address: Optional[str] = Field(default=None, description="Ray cluster address. None for local mode.")
    num_cpus: Optional[int] = Field(default=None, description="Number of CPUs to allocate. None for auto-detection.")
    num_gpus: Optional[int] = Field(default=0, description="Number of GPUs to allocate.")
    memory: Optional[int] = Field(default=None, description="Memory per worker in MB. None for default.")
    object_store_memory: Optional[int] = Field(default=None, description="Object store memory in MB. None for default.")
    redis_max_memory: Optional[int] = Field(default=None, description="Max memory for Redis in MB. None for default.")


class OptimizerConfig(BaseModel):
    """
    Aggregated configuration for all optimizers.
    """
    optuna: OptunaConfig = OptunaConfig()
    ray: RayConfig = RayConfig()
