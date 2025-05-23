# GNNxLLM for Entity Typing
Author: [Denis Fouchard](mailto:denis.fouchard@telecom-paris.fr), LTCI Télécom Paris.

This repository hosts code for my Master Thesis project (PRIM) at Télécom Paris/LTCI, supervised by [Mehwish Alam](https://sites.google.com/view/mehwish-alam/home).

Final Grade : A - GPA 4.0/4.0

The codebase is a heavily modified fork of the great work done by [Xiaoxin He et al](https://arxiv.org/abs/2402.07630). The original repository can be found [here](https://github.com/XiaoxinHe/G-Retriever?tab=readme-ov-file).

## Dependencies
Requires Python 3.11 or later.

```
python -m venv venv
pip install -r requirements
```
## Data Preparation
```
python -m src.dataset.indexing.dbpedia
python -m src.dataset.dbpedia
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run
If you want to train a model with a Bert-like model, you should use the `train_bert.py` script instead ! 

### 1) GraphLLM - Frozen LLM
```
python train.py --dataset dbpedia --model_name graph_llm --frozen_llm True ...
```

### 2) GraphLLM - Tuned LLM
```
python train.py --dataset dbpedia --model_name graph_llm_classifier --llm_frozen False ...
```
### 3) GNN Classifier
```
python train_gnn.py --dataset dbpedia ...
```

For more options during training, see `src/config.py`.

## Available Models
See `src/model/__init__.py` for available models.
- GraphLLM : LLM with GNN encoder, no classification head
- GraphLLMClassifier : LLM with GNN encoder, classification head
- GraphBertClassifier : Bert-like model with GNN encoder
- LLMClassifier : LLM with classification head only

