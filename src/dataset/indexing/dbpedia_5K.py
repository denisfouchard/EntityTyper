# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

MODELNAME = "sbert"
CSV_PATH = (
    "/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia_baseline/DBPediaQA10K.csv"
)
PATH = "dataset/dbpedia_baseline"
path_nodes = f"{PATH}/nodes"
path_edges = f"{PATH}/edges"
path_graphs = f"{PATH}/graphs"


# %%
def clean_dataset():
    print("Cleaning dataset...")
    # Remove samples with empty graphs
    raw_dataset = pd.read_csv(CSV_PATH)
    cleaned_dataset = raw_dataset.copy()
    sub_g_list = []
    for i in tqdm(range(len(raw_dataset))):
        triples_list: str = raw_dataset.iloc[i]["graph"]
        # Deserialize the triples as a list of tuples
        triples_list: list = eval(triples_list)  # type: ignore

        if len(triples_list) == 0:
            # Skip if there are no triples
            # Remove the corresponding row from the dataset
            cleaned_dataset.drop(index=i, inplace=True)
        else:
            sub_g_list.append(triples_list)
    # Rebuild index
    cleaned_dataset.reset_index(drop=True, inplace=True)
    print("Number of samples after cleaning: ", len(cleaned_dataset))
    # Convert the dataframe to a PyTorch dataset
    dataset = Dataset.from_pandas(cleaned_dataset)
    # Save the dataset to disk
    torch.save(dataset, f"{PATH}/sampled_dataset.pt")
    return cleaned_dataset, sub_g_list


def save_nodes_edges_from_dataset(
    dataset: pd.DataFrame, graphs_list: list[list[tuple[str, str, str]]]
):
    print("Saving nodes and edges...")
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    # Rebuild index
    dataset.reset_index(drop=True, inplace=True)
    print("Number of samples after cleaning: ", len(dataset))

    for i in tqdm(range(len(graphs_list))):
        nodes = {}
        edges = []
        triples_list = graphs_list[i]
        for h, r, t in triples_list:
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

    return None


def split_dataset(dataset: pd.DataFrame, train_p=0.7, val_p=0.15):
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[: int(n * train_p)]
    val_indices = indices[int(n * train_p) : int(n * (train_p + val_p))]
    test_indices = indices[int(n * (train_p + val_p)) :]

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


def encode_questions(model, tokenizer, device, text2embedding, dataset: pd.DataFrame):
    questions = dataset["question"]
    questions = questions.tolist()
    print("Number of questions: ", len(questions))
    print("Sample question: ", questions[0])
    # encode questions
    print("Encoding questions...")
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f"{PATH}/q_embs.pt")


def encode_graphs(model, tokenizer, device, text2embedding, dataset: pd.DataFrame):

    print("Encoding graphs...")
    os.makedirs(path_graphs, exist_ok=True)

    for index in tqdm(range(len(dataset))):
        # Check if the graph is already encoded
        if os.path.exists(f"{path_graphs}/{index}.pt"):
            continue
        nodes = pd.read_csv(f"{path_nodes}/{index}.csv")
        edges = pd.read_csv(f"{path_edges}/{index}.csv")

        nodes.node_attr.fillna("", inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # Encode edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
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
    model, tokenizer, device = load_model[MODELNAME]()
    text2embedding = load_text2embedding[MODELNAME]
    dataset, sub_g_list = clean_dataset()
    save_nodes_edges_from_dataset(dataset=dataset, graphs_list=sub_g_list)
    encode_questions(model, tokenizer, device, text2embedding, dataset)
    encode_graphs(model, tokenizer, device, text2embedding, dataset)
    split_dataset(dataset=dataset)
