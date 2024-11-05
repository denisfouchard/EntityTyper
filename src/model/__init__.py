from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "3-8b": "meta-llama/Meta-Llama-3-8B",
    "trained_dbpedia": "/home/infres/dfouchard-21/G-Retriever/output/dbpedia/model_name_graph_llm_llm_model_name_3-8b_llm_frozen_False_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth",
}
