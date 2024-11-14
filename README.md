# G-Retriever

## Data Preparation
```
python -m src.dataset.indexing.dbpedia
python -m src.dataset.dbpedia
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run


### 1) Frozen LLM + Prompt Tuning
```
# prompt tuning
python train.py --dataset scene_graphs_baseline --model_name pt_llm

# G-Retriever
python train.py --dataset scene_graphs --model_name graph_llm
```

### 2) Tuned LLM
```
python train.py --dataset dbpedia --model_name graph_llm_classifier --llm_frozen False
```
