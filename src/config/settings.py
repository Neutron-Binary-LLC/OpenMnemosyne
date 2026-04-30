import os
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # System Settings
    PROJECT_NAME: str = "OpenMnemosyne"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Model Settings
    BASE_SIGNAL_MODEL_TYPE: str = "tft"  # tft or lstm-transformer
    NEURAL_CRITIC_PARAMS: int = 15000000 # ~15M params
    CORRECTIVE_VECTOR_DIM: int = 64
    
    # Training Settings
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    PPO_EPOCHS: int = 10
    GAMMA: float = 0.99
    
    # Reward Weights
    W1_ANN_RETURN: float = 1.0
    W2_DOWNSIDE: float = 0.5
    W3_DIFF_RETURN: float = 0.3
    W4_SANITY_BONUS: float = 0.2
    W5_CRITIC_ALIGNMENT: float = 0.4
    
    # Memory Settings
    VECTOR_DB_URL: str = "http://qdrant:6333"
    RELATIONAL_DB_URL: str = "sqlite:///./mem_palace.db"
    
    # LLM Settings
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_MODEL: str = "gpt-4-turbo" # or custom
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
