import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

MODELNAME = "sbert"
PATH = "dataset/webqsp"
path_nodes = f"{PATH}/nodes"
path_edges = f"{PATH}/edges"
path_graphs = f"{PATH}/graphs"


def save_nodes_edges_from_dataset():
    dataset = load_dataset("rmanluo/RoG-webqsp")
    concat_dataset = concatenate_datasets(
        [
            dataset["train"],
            dataset["validation"],
            dataset["test"],
        ]
    )
    # Sample a small subset of the dataset
    concat_dataset = concat_dataset.shuffle(seed=42).select(range(100))

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(concat_dataset))):
        nodes = {}
        edges = []
        for tri in concat_dataset[i]["graph"]:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({"src": nodes[h], "edge_attr": r, "dst": nodes[t]})
        nodes = pd.DataFrame(
            [{"node_id": v, "node_attr": k} for k, v in nodes.items()],
            columns=["node_id", "node_attr"],
        )
        edges = pd.DataFrame(edges, columns=["src", "edge_attr", "dst"])

        nodes.to_csv(f"{path_nodes}/{i}.csv", index=False)
        edges.to_csv(f"{path_edges}/{i}.csv", index=False)

    # Save the dataset to disk
    torch.save(dataset, f"{PATH}/sampled_dataset.pt")
    torch.save(concat_dataset, f"{PATH}/sampled_concatenated_dataset.pt")
    return dataset, concat_dataset


def split_dataset(dataset=None):
    if dataset is None:
        dataset = load_dataset("rmanluo/RoG-webqsp")

    indices = np.arange(100)
    train_indices = indices[:70]
    val_indices = indices[70:85]
    test_indices = indices[85:]

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f"{PATH}/split", exist_ok=True)

    # Save the indices to separate files
    with open(f"{PATH}/split/train_indices.txt", "w") as file:
        file.write("\n".join(map(str, train_indices)))

    with open(f"{PATH}/split/val_indices.txt", "w") as file:
        file.write("\n".join(map(str, val_indices)))

    with open(f"{PATH}/split/test_indices.txt", "w") as file:
        file.write("\n".join(map(str, test_indices)))


def _encode_graph(index, model, tokenizer, device, text2embedding):
    nodes = pd.read_csv(f"{path_nodes}/{index}.csv")
    print("Shape of nodes: ", nodes.shape)
    edges = pd.read_csv(f"{path_edges}/{index}.csv")
    print("Shape of edges: ", edges.shape)
    print(nodes.node_attr)
    nodes.node_attr.fillna("", inplace=True)
    x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
    print("Shape of x: ", x.shape)

    # Encode edges
    edge_attr = text2embedding(
        model, tokenizer, device, edges.edge_attr.tolist()
    )
    edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])
    print("Shape of edge_attr: ", edge_attr.shape)
    print("Shape of edge_index: ", edge_index.shape)
    pyg_graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes),
    )
    if pyg_graph is not None:
        torch.save(pyg_graph, f"{path_graphs}/{index}.pt")


def encode_graphs(model, tokenizer, device, text2embedding):
    print("Encoding graphs...")
    os.makedirs(path_graphs, exist_ok=True)

    for index in tqdm(range(100)):
        # Check if the graph is already encoded
        if os.path.exists(f"{path_graphs}/{index}.pt"):
            continue
        nodes = pd.read_csv(f"{path_nodes}/{index}.csv")
        edges = pd.read_csv(f"{path_edges}/{index}.csv")

        nodes.node_attr.fillna("", inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # Encode edges
        edge_attr = text2embedding(
            model, tokenizer, device, edges.edge_attr.tolist()
        )
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes),
        )
        if pyg_graph is not None:
            torch.save(pyg_graph, f"{path_graphs}/{index}.pt")


if __name__ == "__main__":
    # dataset, concat_dataset = save_nodes_edges_from_dataset()
    model, tokenizer, device = load_model[MODELNAME]()
    text2embedding = load_text2embedding[MODELNAME]
    encode_graphs(model, tokenizer, device, text2embedding)
    # split_dataset(concat_dataset)
