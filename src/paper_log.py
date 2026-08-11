import json, time, os

def log_paper_result(label, value, notes="", drive_path='/content/drive/MyDrive/tool_call_grounding/paper_results_log.jsonl',
                       local_path='/content/paper_results_log.jsonl'):
    """
    Appends one finalized result to TWO files: the Drive copy (persistent,
    shared across sessions) and a local Colab copy (downloadable directly,
    so you have a portable backup independent of Drive/Kaggle).
    """
    entry = {"label": label, "value": value, "notes": notes, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
    line = json.dumps(entry) + '\n'

    with open(drive_path, 'a') as f:
        f.write(line)
    with open(local_path, 'a') as f:
        f.write(line)

    print(f"Logged: {label} = {value}")
