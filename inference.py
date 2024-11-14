import os
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import llama_model_path
from src.model.graph_llm_classifier import GraphLLMClassifier
from src.dataset.dbpedia import DBPediaDataset
from src.dataset.utils.mapping import generate_hierarchical_mapping
from src.utils.collate import collate_fn

# Define the number of classes, and do one-hot encoding for the labels
class2idx, idx2class = generate_hierarchical_mapping(
    file_path="/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia/hierarchy_ids.txt"
)
n_classes = len(idx2class)


def main(args):
    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(
        project=f"{args.project}",
        name=f"{args.dataset}_{args.model_name}_seed{seed}",
        config=args,
    )

    seed_everything(seed=seed)

    dataset = DBPediaDataset()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    test_dataset = [dataset[i] for i in idx_split["test"]]
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    args.llm_frozen = "True"
    model = GraphLLMClassifier.from_pretrained(
        args=args, n_classes=n_classes, model_path=args.checkpoint_path
    )

    # Step 4. Evaluating
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    acc = 0
    for _, batch in enumerate(test_loader):
        y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(model.device)
        with torch.no_grad():
            y_pred = model.predict(batch)
            batch_acc = (y_pred == y_true).sum().item()
            acc += batch_acc

        print(f"Batch Acc {batch_acc/len(y_true)}")

        progress_bar_test.update(1)

    acc /= len(test_loader.dataset)
    print(f"Test Acc {acc}")
    wandb.log({"Test Acc": acc})


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
