"""
MongoDB database connection and configuration.
"""
import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MongoDBConfig:
    """MongoDB configuration from environment variables."""
    
    def __init__(self):
        self.uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database: str = os.getenv("MONGODB_DATABASE", "swiss_cv_generator")
        self.username: Optional[str] = os.getenv("MONGODB_USERNAME") or None
        self.password: Optional[str] = os.getenv("MONGODB_PASSWORD") or None
        self.auth_source: str = os.getenv("MONGODB_AUTH_SOURCE", "admin")
    
    def get_connection_string(self) -> str:
        """
        Build MongoDB connection string.
        
        Returns:
            MongoDB connection URI string.
        """
        # If URI already contains authentication, use it as-is
        if "@" in self.uri or "mongodb+srv://" in self.uri:
            return self.uri
        
        # Otherwise, build connection string with credentials if provided
        if self.username and self.password:
            # Parse the URI and add credentials
            if self.uri.startswith("mongodb://"):
                # mongodb://host:port -> mongodb://user:pass@host:port
                uri_without_protocol = self.uri.replace("mongodb://", "")
                return f"mongodb://{self.username}:{self.password}@{uri_without_protocol}"
            elif self.uri.startswith("mongodb+srv://"):
                # mongodb+srv://host -> mongodb+srv://user:pass@host
                uri_without_protocol = self.uri.replace("mongodb+srv://", "")
                return f"mongodb+srv://{self.username}:{self.password}@{uri_without_protocol}"
        
        return self.uri


# Singleton MongoDB client instance
_mongodb_client: Optional[MongoClient] = None
_mongodb_db: Optional[Database] = None


def get_mongodb_client() -> MongoClient:
    """
    Get or create MongoDB client singleton instance.
    
    Returns:
        MongoClient instance.
    
    Raises:
        ConnectionFailure: If connection to MongoDB fails.
        ConfigurationError: If MongoDB configuration is invalid.
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        config = MongoDBConfig()
        connection_string = config.get_connection_string()
        
        try:
            _mongodb_client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection
            _mongodb_client.admin.command('ping')
        except ConnectionFailure as e:
            raise ConnectionFailure(
                f"Failed to connect to MongoDB at {connection_string}: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Invalid MongoDB configuration: {e}"
            ) from e
    
    return _mongodb_client


def get_mongodb_database() -> Database:
    """
    Get MongoDB database instance.
    
    Returns:
        Database instance.
    
    Raises:
        ConnectionFailure: If connection to MongoDB fails.
    """
    global _mongodb_db
    
    if _mongodb_db is None:
        config = MongoDBConfig()
        client = get_mongodb_client()
        _mongodb_db = client[config.database]
    
    return _mongodb_db


def close_mongodb_connection():
    """Close MongoDB connection."""
    global _mongodb_client, _mongodb_db
    
    if _mongodb_client:
        _mongodb_client.close()
        _mongodb_client = None
        _mongodb_db = None


def test_connection() -> bool:
    """
    Test MongoDB connection.
    
    Returns:
        True if connection is successful, False otherwise.
    """
    try:
        client = get_mongodb_client()
        client.admin.command('ping')
        return True
    except Exception:
        return False

