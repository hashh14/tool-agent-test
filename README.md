# Tool-Call Argument Grounding Detection

Detects hallucinated (ungrounded) arguments in LLM agent tool calls at the
individual-argument level, using internal model activations.

## Pipeline
1. `data_processing.py` - benchmark construction from Glaive function-calling data
   (parsing, span-finding, context corruption)
2. `model_utils.py` - model loading and activation extraction
3. `crosscoder.py` - shared-dictionary crosscoder for correlational feature analysis
4. `intervention.py` - causal intervention/activation-patching utilities
5. `classifier.py` - mutual-information feature ranking, classifier training, and setup helpers
6. `baseline.py` - reimplementation of Healy et al. (2601.05214) whole-call baseline
7. `inference.py` - deployable, single-forward-pass grounding checker (main entry point)

## Usage
See `inference.py`'s `check_tool_call_grounding` for the primary inference function.

## Data
Built from the Glaive Function-Calling v2 dataset. Benchmark construction and
label integrity checks documented in project notebooks.
