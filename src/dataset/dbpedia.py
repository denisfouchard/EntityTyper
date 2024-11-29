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
from src.dataset.utils.mapping import generate_mapping, generate_hierarchical_mapping

# Remove warnings
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DBPediaDataset(Dataset):
    """
    A dataset class for handling DBPedia data with optional PCST retrieval.

    Attributes:
        path (str): The base path to the dataset.
        nodes_dir (str): Directory containing node files.
        edges_dir (str): Directory containing edge files.
        graphs_dir (str): Directory containing graph files.
        retrieval (bool): Flag indicating whether to perform PCST retrieval.
        version (str): Version of the dataset.
        cached_graph_dir (str): Directory for cached graphs after retrieval.
        cached_desc_dir (str): Directory for cached descriptions after retrieval.
        prompt (str): Default prompt for questions.
        graph (None): Placeholder for graph data.
        graph_type (str): Type of graph used, default is "Knowledge Graph".
        dataset (torch.Tensor): Loaded dataset tensor.
        q_embs (torch.Tensor): Loaded question embeddings tensor.

    Methods:
        __post_init__(): Initializes the dataset and performs PCST retrieval if enabled.
        __len__(): Returns the length of the dataset.
        __getitem__(index): Retrieves the data sample at the specified index.
        get_idx_split(): Returns the indices for train, validation, and test splits.
    """

    def __init__(self, version: str = "", retrieval: bool = True, summary: bool = True):
        super().__init__()
        self.path = f"/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia{version}"
        self.nodes_dir = f"{self.path}/nodes"
        self.edges_dir = f"{self.path}/edges"
        self.graphs_dir = f"{self.path}/graphs"
        self.retrieval = retrieval
        self.version = version

        self.cached_graph_dir = ""
        self.cached_desc_dir = ""
        self.class2idx, self.idx2class = generate_hierarchical_mapping(
            f"{self.path}/hierarchy_ids.txt"
        )
        self.classes = list(self.class2idx.keys())
        self.prompt = open(f"{self.path}/prompt", "r").read()
        self.graph = None
        self.graph_type = "Knowledge Graph"
        self.dataset = torch.load(
            f"{self.path}/sampled_dataset.pt", map_location=device
        )
        self.q_embs = torch.load(f"{self.path}/q_embs.pt", map_location=device)

        if summary:
            self.summary()

    def __post_init__(self):
        """
        Post-initialization method for the DBPedia dataset class.

        This method performs the Prize-Collecting Steiner Tree (PCST) retrieval if the `retrieval` attribute is set to True.
        It creates directories for cached graphs and descriptions, and processes each query embedding to retrieve subgraphs
        and descriptions, which are then cached for future use.

        If the `retrieval` attribute is False, it simply prints a message indicating that the dataset has been initialized
        without retrieval.

        Attributes:
            retrieval (bool): Flag indicating whether to perform PCST retrieval.
            path (str): Base path for the dataset.
            cached_graph_dir (str): Directory to cache retrieved subgraphs.
            cached_desc_dir (str): Directory to cache retrieved descriptions.
            q_embs (list): List of query embeddings.
            graphs_dir (str): Directory containing graph files.
            nodes_dir (str): Directory containing node files.
            edges_dir (str): Directory containing edge files.
            version (str): Version of the DBPedia dataset.

        Raises:
            FileNotFoundError: If the graph, node, or edge files are not found.
        """
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

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.dataset)

    def summary(self):
        """
        Print a summary of the dataset.
        """
        rep_classes = 0
        print(f"==========[DBPedia{self.version} Dataset Summary]==========")
        print(f"Number of samples: {len(self.dataset)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Number of query embeddings: {len(self.q_embs)}")
        print(f"Graph type: {self.graph_type}")
        print("Number of samples per class:")
        n_sample_per_class = {c: 0 for c in self.classes}
        for item in self.dataset:
            n_sample_per_class[item["answer"]] += 1
        for c, n in n_sample_per_class.items():
            if n > 0:
                rep_classes += 1
            print(f"  - {c}: {n}")
        print(
            f"Number of classes represented: {rep_classes} ({100*rep_classes/len(self.classes)}%)"
        )

    def __getitem__(self, index) -> dict:
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "id" (int): The index of the item.
                - "question" (str): The formatted question string.
                - "label" (str): The answer label.
                - "graph" (torch.Tensor): The graph data loaded from a file.
                - "desc" (str): The description text loaded from a file (empty if retrieval is False).
        """
        item = self.dataset[index]
        entity = item["q_entity"]
        question = f'## Instructions\n {self.prompt}\n ## Question\n {item["question"]}'
        if self.retrieval:
            graph = torch.load(f"{self.cached_graph_dir}/{index}.pt")
            desc = open(
                f"{self.cached_desc_dir}/{index}.txt",
                "r",
            ).read()
        else:
            graph = torch.load(f"{self.graphs_dir}/{index}.pt")
            desc = ""
        label = item["answer"]

        return {
            "id": index,
            "question": question,
            "entity": entity,
            "label": label,
            "graph": graph,
            "desc": desc,
        }

    def get_idx_split(self) -> dict[str, list[int]]:
        """
        Reads and returns the indices for train, validation, and test splits from text files.

        The method expects the following files to be present in the `self.path/split/` directory:
        - train_indices.txt
        - val_indices.txt
        - test_indices.txt

        Each file should contain one index per line.

        Returns:
            dict[str, list[int]]: A dictionary with three keys: 'train', 'val', and 'test'.
            Each key maps to a list of integers representing the indices for the respective split.
        """
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
    parser.add_argument("--retrieval", type=bool, default=False)
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
