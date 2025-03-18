# ./settings/base_settings.py

from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    """
    Base configuration class that sets the environment file for all configurations.

    Attributes:
        Config: Internal Pydantic configuration class to specify the environment file.
    """

    class Config:
        """
        Configuration for BaseConfig.

        Specifies the `.env` file to load environment variables from.
        """
        env_file = ".env"
        extra = "ignore"