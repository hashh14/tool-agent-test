import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    return model, tokenizer


def build_continuation(function_name, arguments):
    call_dict = {"name": function_name, "arguments": arguments}
    continuation = ' <functioncall> ' + json.dumps(call_dict)
    return continuation


def get_target_token_position(tokenizer, prompt_text, continuation_text, target_value):
    full_text = prompt_text + continuation_text
    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding['offset_mapping'][0]

    value_json = json.dumps(target_value)
    idx_in_continuation = continuation_text.find(value_json)
    if idx_in_continuation == -1:
        return None, None

    char_start = len(prompt_text) + idx_in_continuation
    char_end = char_start + len(value_json)

    token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < char_end and e > char_start]
    if not token_positions:
        return None, None
    return encoding, token_positions


def extract_activation(model, tokenizer, prompt_text, continuation_text, target_value, layer=12):
    encoding, token_positions = get_target_token_position(tokenizer, prompt_text, continuation_text, target_value)
    if encoding is None:
        return None
    input_ids = encoding['input_ids'].to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    layer_activations = outputs.hidden_states[layer][0]
    target_vector = layer_activations[token_positions].mean(dim=0)
    return target_vector.cpu()
