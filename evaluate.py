import os
import torch
import wandb
import gc
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from src.utils.seeding import seed_everything
from src.config import parse_args_llama
from src.model import llama_model_path
from src.model.graph_llm_classifier import GraphLLMClassifier
from src.dataset.dbpedia import DBPediaDataset
from src.utils.collate import collate_fn

# Define the number of classes, and do one-hot encoding for the labels


def main(args, path=None, test_loader=None, model=None, checkpoint_path=None):
    # Step 1: Set up wandb
    seed = args.seed
    name = f"DBPedia60K - No GNN - Frozen LLM"
    wandb.init(
        project=f"{args.project}",
        name=name,
        config=args,
    )

    seed_everything(seed=seed)
    retrieval = args.retrieval == "True"
    if not test_loader:
        dataset = DBPediaDataset(retrieval=retrieval, version="60k")
        class2idx, idx2class = dataset.class2idx, dataset.idx2class
        n_classes = len(idx2class)
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
    if not model:
        args.llm_model_path = llama_model_path[args.llm_model_name]
        args.llm_frozen = "True"
        if not checkpoint_path:
            checkpoint_path = args.checkpoint_path
        model = GraphLLMClassifier.from_pretrained(
            args=args, n_classes=n_classes, model_path=checkpoint_path
        )

    # Step 4. Evaluating
    os.makedirs(f"{args.output_dir}/{args.dataset}/evaluation/", exist_ok=True)
    if path is None:
        path = f"{args.output_dir}/{args.dataset}/evaluation/Results-{name.replace(" ", "")}.csv"
    print(f"Results saving path: {path}")
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    test_accuracies = []
    true_labels = []
    pred_labels = []
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            # Append the list of true labels to the dataframe column
            true_labels.extend([l for l in batch["label"]])
            y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(
                model.device
            )

            y_pred = model.predict(batch)
            batch_accuracy = torch.mean((y_pred == y_true).float())
            test_accuracies.append(batch_accuracy.item())

        pred_labels.extend([l for l in [idx2class[i] for i in y_pred]])
        progress_bar_test.set_postfix_str(
            f"Acc: {sum(test_accuracies) / len(test_accuracies)}"
        )

        progress_bar_test.update(1)

    acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Test Accuracy {acc}")
    wandb.log({"Test Acc": acc})

    # Save the results to a csv file
    results = pd.DataFrame({"true_labels": true_labels, "pred_labels": pred_labels})
    results.to_csv(path, index=False)


if __name__ == "__main__":

    args = parse_args_llama()

    main(
        args=args,
        checkpoint_path="/home/infres/dfouchard-21/G-Retriever/output/dbpedia60k/DBPedia60K-NoGNN-FrozenLLM_llm_frozen_True_10_epochs_checkpoint_best.pth",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
