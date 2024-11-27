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
        name=f"DBPedia12K - No Retrieval",
        config=args,
    )

    seed_everything(seed=seed)
    retrieval = args.retrieval == "True"
    dataset = DBPediaDataset(retrieval=retrieval)
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
    # Step 4. Evaluating
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    path = f"{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv"
    print(f"Results saving path: {path}")
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    test_accuracies = []
    true_labels = []
    pred_labels = []
    for _, batch in enumerate(test_loader):
        # Append the list of true labels to the dataframe column
        true_labels.extend([l for l in batch["label"]])
        y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(model.device)
        with torch.no_grad():
            y_pred = model.predict(batch)
            batch_accuracy = torch.mean((y_pred == y_true).float())
            test_accuracies.append(batch_accuracy.item())

        pred_labels.extend([l for l in [idx2class[i] for i in y_pred]])

        progress_bar_test.update(1)

    acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Test Acc {acc}")
    wandb.log({"Test Acc": acc})

    # Save the results to a csv file
    results = pd.DataFrame({"true_labels": true_labels, "pred_labels": pred_labels})
    results.to_csv(path, index=False)


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
