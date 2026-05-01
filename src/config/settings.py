import os
import yaml
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

def load_config(config_path: str = "config.yaml") -> Dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config_data = load_config()
reward_weights = config_data.get("reward_weights", {})

class Settings(BaseSettings):
    # System Settings
    PROJECT_NAME: str = config_data.get("project_name", "OpenMnemosyne")
    ENV: str = config_data.get("env", "development")
    LOG_LEVEL: str = config_data.get("log_level", "INFO")
    
    # Model Settings
    DEVICE: str = config_data.get("device") or ("cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu")
    BASE_SIGNAL_MODEL_TYPE: str = config_data.get("base_signal_model_type", "tft")
    NEURAL_CRITIC_PARAMS: int = config_data.get("neural_critic_params", 15000000)
    CORRECTIVE_VECTOR_DIM: int = config_data.get("corrective_vector_dim", 64)
    
    # Training Settings
    LEARNING_RATE: float = config_data.get("learning_rate", 1e-4)
    BATCH_SIZE: int = config_data.get("batch_size", 64)
    PPO_EPOCHS: int = config_data.get("ppo_epochs", 10)
    GAMMA: float = config_data.get("gamma", 0.99)
    
    # Reward Weights
    W1_ANN_RETURN: float = reward_weights.get("w1_ann_return", 1.0)
    W2_DOWNSIDE: float = reward_weights.get("w2_downside", 0.5)
    W3_DIFF_RETURN: float = reward_weights.get("w3_diff_return", 0.3)
    W4_SANITY_BONUS: float = reward_weights.get("w4_sanity_bonus", 0.2)
    W5_CRITIC_ALIGNMENT: float = reward_weights.get("w5_critic_alignment", 0.4)
    
    # Memory Settings
    VECTOR_DB_URL: str = config_data.get("vector_db_url", "http://qdrant:6333")
    RELATIONAL_DB_URL: str = config_data.get("relational_db_url", "sqlite:///./mem_palace.db")
    
    # LLM Settings
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_MODEL: str = config_data.get("llm_model", "gpt-4-turbo")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
