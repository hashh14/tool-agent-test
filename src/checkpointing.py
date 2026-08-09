import os
import pickle
import json
import time

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {'processed_indices': set(), 'results': []}

def save_checkpoint(state, path):
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def log_event(message, log_path='/content/drive/MyDrive/tool_call_grounding/logs/run_log.jsonl'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'message': str(message)}
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(message)
