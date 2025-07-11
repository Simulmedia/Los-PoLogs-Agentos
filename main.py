import os
from config.settings import API_KEY_FILEPATH, DEFAULT_MAX_WORKERS, OUTPUT_DATA_FILE, INPUT_DATA_FILE, \
    DEFAULT_TEMPERATURE, DEFAULT_BATCH_SIZE
from src.log_normalization_assistant import DataOrchestrator

if __name__ == '__main__':

    if not os.path.exists(API_KEY_FILEPATH):
        print(f"Error: API key file not found at '{API_KEY_FILEPATH}'")
    elif not os.path.exists(INPUT_DATA_FILE):
        print(f"Error: Data file not found at '{INPUT_DATA_FILE}'. Please check the path.")
    else:
        orchestrator = DataOrchestrator(filepath=INPUT_DATA_FILE, api_key_path=API_KEY_FILEPATH, max_workers=DEFAULT_MAX_WORKERS, batch_size=DEFAULT_BATCH_SIZE, temperature=DEFAULT_TEMPERATURE)
        final_output = orchestrator.run()
        final_output.to_excel(OUTPUT_DATA_FILE, index=False)


