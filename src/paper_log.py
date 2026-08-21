import json, time, os
import numpy as np

def _make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return obj

def log_result(filename, label, value, notes=""):
    """
    Generic logger. filename: e.g. 'paper_results', 'bigmodel_results',
    'exploratory_results', 'rigor_results' - no path/extension needed.
    Writes to both Drive (persistent) and a local Colab copy (downloadable).
    """
    drive_path = f'/content/drive/MyDrive/tool_call_grounding/{filename}_log.jsonl'
    local_path = f'/content/{filename}_log.jsonl'
    safe_value = _make_json_safe(value)
    entry = {"label": label, "value": safe_value, "notes": notes, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
    line = json.dumps(entry) + '\n'
    with open(drive_path, 'a') as f:
        f.write(line)
    with open(local_path, 'a') as f:
        f.write(line)
    print(f"[{filename}] Logged: {label} = {safe_value}")
