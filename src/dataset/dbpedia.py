# %%
import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# Add the path to the sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils.retrieval import PCST_retrieve

# Remove warnings
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DBPediaDataset(Dataset):
    def __init__(self, version: str = "", retrieval: bool = True):
        super().__init__()
        self.path = f"/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia{version}"
        self.nodes_dir = f"{self.path}/nodes"
        self.edges_dir = f"{self.path}/edges"
        self.graphs_dir = f"{self.path}/graphs"
        self.retrieval = retrieval
        self.version = version

        self.cached_graph_dir = ""
        self.cached_desc_dir = ""

        self.prompt = "Please answer the given question."
        self.graph = None
        self.graph_type = "Knowledge Graph"
        self.dataset = torch.load(
            f"{self.path}/sampled_dataset.pt", map_location=device
        )
        self.q_embs = torch.load(f"{self.path}/q_embs.pt", map_location=device)

    def __post_init__(self):
        # Perform PCST retrieval if retrieval is True
        if self.retrieval:
            self.cached_graph_dir = f"{self.path}/cached_graphs"
            self.cached_desc_dir = f"{self.path}/cached_desc"
            os.makedirs(self.cached_desc_dir, exist_ok=True)
            os.makedirs(self.cached_graph_dir, exist_ok=True)
            print("Performing PCST retrieval...")
            for index in tqdm(range(len(self.q_embs))):
                if os.path.exists(f"{self.cached_graph_dir}/{index}.pt"):
                    continue
                graph = torch.load(f"{self.graphs_dir}/{index}.pt")
                nodes = pd.read_csv(f"{self.nodes_dir}/{index}.csv")
                edges = pd.read_csv(f"{self.edges_dir}/{index}.csv")
                q_emb = self.q_embs[index]
                subg, desc = PCST_retrieve(
                    graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5
                )
                torch.save(subg, f"{self.cached_graph_dir}/{index}.pt")
                open(f"{self.cached_desc_dir}/{index}.txt", "w").write(desc)

            print(f"Initialized DBPedia{self.version} dataset with PCST retrieval.")
        else:
            print(f"Initialized DBPedia{self.version} dataset with no retrieval.")

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        if self.retrieval:
            graph = torch.load(f"{self.cached_graph_dir}/{index}.pt")
            desc = open(f"{self.cached_desc_dir}/{index}.txt", "r").read()
        else:
            graph = torch.load(f"{self.graphs_dir}/{index}.pt")
            desc = ""
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
        with open(f"{self.path}/split/train_indices.txt", "r") as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f"{self.path}/split/val_indices.txt", "r") as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f"{self.path}/split/test_indices.txt", "r") as file:
            test_indices = [int(line.strip()) for line in file]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--retrieval", type=bool, default=True)
    args = parser.parse_args()

    version = args.version
    retrieval = args.retrieval
    # Create the cached directories
    dbpedia = DBPediaDataset(version=version, retrieval=retrieval)
    # %%
    data = dbpedia[1]
    for k, v in data.items():
        print(f"{k}: {v}")

    split_ids = dbpedia.get_idx_split()
    for k, v in split_ids.items():
        print(f"# {k}: {len(v)}")

# %%
