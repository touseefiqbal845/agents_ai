# Configuration file for Multi-Modal Agents

# API Keys
OPENAI_API_KEY = "Kt9cpecoNyy3YtdAVgV5UrHCxeBc6SDF5mxrMGfz40QjLvpw6UKo9E9WMnGY6hQA"
CHROMA_API_KEY = "ck-CbZnvFgNiZrbkDZt2JA4"
CHROMA_TENANT = "96b80992-35166049b90112f"
CHROMA_DB_NAME = "tousvector_db"

# Database Settings
DATABASE_NAME = "agent_data.db"
LOG_FILE = "agent_activity.log"

# OpenAI Settings
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
DALL_E_SIZE = "1024x1024"

# Memory Settings
MEMORY_COLLECTION_NAME = "long_term_memory_new"
DEFAULT_MEMORY_RESULTS = 3

# Processing Settings
MAX_SUMMARY_LENGTH = 200
DEFAULT_TRANSLATION_LANGUAGE = "es"
DEFAULT_CODE_LANGUAGE = "python"

# File Settings
BACKUP_DIR_PREFIX = "backup_"
MAX_FILE_SEARCH_RESULTS = 10

# System Settings
SCHEDULER_SLEEP_TIME = 1
DEFAULT_PRIORITY = "medium"
DEFAULT_TASK_STATUS = "pending"

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Feature Flags
ENABLE_IMAGE_GENERATION = True
ENABLE_FILE_OPERATIONS = True
ENABLE_SYSTEM_MONITORING = True
ENABLE_ANALYTICS = True
ENABLE_BACKUP = True
ENABLE_ENCRYPTION = True
