from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


# --- OpenAI API Configuration ---
API_KEY_FILEPATH = PROJECT_DIR  / 'config' / 'openai_api_key.txt'
PRIMARY_MODEL = "gpt-4.1-mini"
SECONDARY_MODEL = "o4-mini"

# --- File Paths ---
INPUT_DATA_FILE = PROJECT_DIR / 'data' / 'input_data' / 'Interactive Brokers Post Logs.5.12.xlsx'
OUTPUT_DATA_FILE = PROJECT_DIR / 'data' / 'results' / 'IB.xlsx'

SYSTEM_PROMPT1_PATH = PROJECT_DIR  / 'config' / 'prompts' / 'system_prompt1.txt'
SYSTEM_PROMPT2_PATH = PROJECT_DIR  / 'config' / 'prompts' / 'system_prompt2.txt'
SYSTEM_PROMPT3_PATH = PROJECT_DIR  / 'config' / 'prompts' / 'system_prompt3.txt'

# --- Processing Parameters ---
DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_WORKERS = 16
DEFAULT_TEMPERATURE = 0.1
