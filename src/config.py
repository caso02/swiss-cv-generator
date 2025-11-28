"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Try to use pydantic-settings, fallback to manual loading
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        """Application settings loaded from environment variables."""
        
        # MongoDB Configuration
        mongodb_uri: str = "mongodb://localhost:27017"
        
        # Source Database (read-only, existing scraper DB)
        mongodb_database_source: str = "CV_DATA"
        mongodb_collection_occupations: str = "cv_berufsberatung"
        
        # Target Database (read/write, new app DB)
        mongodb_database_target: str = "swiss_cv_generator"
        
        # OpenAI Configuration
        openai_api_key: Optional[str] = None
        openai_model_mini: str = "gpt-3.5-turbo"
        openai_model_full: str = "gpt-4"
        
        # Application Configuration
        data_dir: str = "data"
        log_level: str = "INFO"
        
        # AI Configuration
        ai_max_retries: int = 5
        ai_rate_limit_delay: float = 1.0
        ai_temperature_creative: float = 0.8
        ai_temperature_factual: float = 0.3
        
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore"
        )
except ImportError:
    # Fallback: Use pydantic BaseSettings directly (for older versions)
    try:
        from pydantic import BaseSettings
        
        class Settings(BaseSettings):
            """Application settings loaded from environment variables."""
            
            # MongoDB Configuration
            mongodb_uri: str = "mongodb://localhost:27017"
            
            # Source Database (read-only, existing scraper DB)
            mongodb_database_source: str = "CV_DATA"
            mongodb_collection_occupations: str = "cv_berufsberatung"
            
            # Target Database (read/write, new app DB)
            mongodb_database_target: str = "swiss_cv_generator"
            
            # OpenAI Configuration
            openai_api_key: Optional[str] = None
            openai_model_mini: str = "gpt-3.5-turbo"
            openai_model_full: str = "gpt-4"
            
            # Application Configuration
            data_dir: str = "data"
            log_level: str = "INFO"
            
            # AI Configuration
            ai_max_retries: int = 5
            ai_rate_limit_delay: float = 1.0
            ai_temperature_creative: float = 0.8
            ai_temperature_factual: float = 0.3
            
            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"
                case_sensitive = False
                extra = "ignore"
    except ImportError:
        # Final fallback: Manual settings class using dotenv
        load_dotenv()
        
        class Settings:
            """Application settings loaded from environment variables."""
            
            def __init__(self):
                # MongoDB Configuration
                self.mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
                
                # Source Database (read-only, existing scraper DB)
                self.mongodb_database_source: str = os.getenv("MONGODB_DATABASE_SOURCE", "CV_DATA")
                self.mongodb_collection_occupations: str = os.getenv("MONGODB_COLLECTION_OCCUPATIONS", "cv_berufsberatung")
                
                # Target Database (read/write, new app DB)
                self.mongodb_database_target: str = os.getenv("MONGODB_DATABASE_TARGET", "swiss_cv_generator")
                
                # OpenAI Configuration
                self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
                self.openai_model_mini: str = os.getenv("OPENAI_MODEL_MINI", "gpt-3.5-turbo")
                self.openai_model_full: str = os.getenv("OPENAI_MODEL_FULL", "gpt-4")
                
                # Application Configuration
                self.data_dir: str = os.getenv("DATA_DIR", "data")
                self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
                
                # AI Configuration
                self.ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "5"))
                self.ai_rate_limit_delay: float = float(os.getenv("AI_RATE_LIMIT_DELAY", "1.0"))
                self.ai_temperature_creative: float = float(os.getenv("AI_TEMPERATURE_CREATIVE", "0.8"))
                self.ai_temperature_factual: float = float(os.getenv("AI_TEMPERATURE_FACTUAL", "0.3"))


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the singleton settings instance.
    
    Returns:
        Settings instance loaded from environment variables and .env file.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Export singleton as 'settings'
settings = get_settings()

