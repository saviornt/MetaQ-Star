from pydantic import BaseModel, Field

class ResourceSettings(BaseModel):
    observation_period: float = Field(
        default=10.0, description="Duration in seconds to observe before pruning."
    )
    check_interval: float = Field(
        default=1.0, description="Interval in seconds between resource usage checks."
    )
    rolling_window_size: int = Field(
        default=20, description="Number of samples to keep for rolling averages."
    )
    target_utilization: float = Field(
        default=0.9, description="Target fraction (~0.9 = 90%) of resource utilization."
    )
    enabled: bool = Field(
        default=False, description="Enable or disable the dynamic resource manager."
    )