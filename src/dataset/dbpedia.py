# %%
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from src.dataset.utils.retrieval import PCST_retrieve

# Remove warnings
import warnings

warnings.filterwarnings("ignore")

PATH = "/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia"
path_nodes = f"{PATH}/nodes"
path_edges = f"{PATH}/edges"
path_graphs = f"{PATH}/graphs"

cached_graph = f"{PATH}/cached_graphs"
cached_desc = f"{PATH}/cached_desc"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DBPediaDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = "Please answer the given question."
        self.graph = None
        self.graph_type = "Knowledge Graph"
        self.dataset = torch.load(f"{PATH}/sampled_dataset.pt", map_location=device)
        self.q_embs = torch.load(f"{PATH}/q_embs.pt", map_location=device)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f"{cached_graph}/{index}.pt")
        desc = open(f"{cached_desc}/{index}.txt", "r").read()
        label = data["answer"]

        return {
            "id": index,
            "question": question,
            "label": label,
            "graph": graph,
            "desc": desc,
        }

    def get_idx_split(self):
        # Load the saved indices
        with open(f"{PATH}/split/train_indices.txt", "r") as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f"{PATH}/split/val_indices.txt", "r") as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f"{PATH}/split/test_indices.txt", "r") as file:
            test_indices = [int(line.strip()) for line in file]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }


# %%
if __name__ == "__main__":

    # Create the cached directories
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    q_embs = torch.load(f"{PATH}/q_embs.pt")
    for index in tqdm(range(len(q_embs))):
        if os.path.exists(f"{cached_graph}/{index}.pt"):
            continue
        graph = torch.load(f"{path_graphs}/{index}.pt")
        nodes = pd.read_csv(f"{path_nodes}/{index}.csv")
        edges = pd.read_csv(f"{path_edges}/{index}.csv")
        q_emb = q_embs[index]
        subg, desc = PCST_retrieve(
            graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5
        )
        torch.save(subg, f"{cached_graph}/{index}.pt")
        open(f"{cached_desc}/{index}.txt", "w").write(desc)

    dbpedia = DBPediaDataset()
    # %%
    data = dbpedia[1]
    for k, v in data.items():
        print(f"{k}: {v}")

    split_ids = dbpedia.get_idx_split()
    for k, v in split_ids.items():
        print(f"# {k}: {len(v)}")

# %%
