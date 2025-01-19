from .gnn_classifier import GNNClassifier
from .llm_classifier import LLMClassifier
from .graph_llm_classifier import GraphLLMClassifier
from .base_classifier import EntityClassifier

# Replace the following with the model paths
llama_model_path = {
    "llama": "meta-llama/Meta-Llama-3-8B",
}


CLASSIFIER_MODEL_MAPPING: dict[str, EntityClassifier] = {
    "gnn": GNNClassifier,
    "llm": LLMClassifier,
    "graph_llm": GraphLLMClassifier,
}
