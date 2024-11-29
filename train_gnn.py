"""Training script for the GraphLLMClassifier model"""

import os
from argparse import Namespace
import gc
import torch
from tqdm import tqdm
from accelerate import infer_auto_device_map, dispatch_model
from torch.amp import autocast
from src.model import llama_model_path
from src.model.gnn_classifier import GNNClassifier
from src.dataset.dbpedia import DBPediaDataset
from src.config import parse_args_llama
from src.utils.checkpoint import save_checkpoint
from src.utils.collate import collate_fn
from src.utils.seeding import seed_everything
from src.utils.lr_scheduling import adjust_learning_rate
from src.dataset.utils.mapping import generate_hierarchical_mapping
from src.dataset.utils.dataloader import dataset_loader
import wandb

# Define the number of classes, and do one-hot encoding for the labels
class2idx, idx2class = generate_hierarchical_mapping(
    file_path="/home/infres/dfouchard-21/G-Retriever/dataset/dbpedia/hierarchy_ids.txt"
)
n_classes = len(idx2class)


# Define the number of classes, and do one-hot encoding for the labels
def batch_one_hot_encode(y_str: list[str], num_classes):
    labels = [class2idx[c] for c in y_str]
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot[range(len(labels)), labels] = 1
    return one_hot


def main(args: Namespace) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    # Step 1: Set up wandb
    wandb.init(
        project=f"{args.project}",
        name=f"DBPedia12K - No Retrieval - SBert+BigMLP",
        config=args,
    )

    seed_everything(seed=args.seed)
    retrieval = args.retrieval == "True"
    dataset = DBPediaDataset(retrieval=retrieval)
    args.model_name = "gnn_classifier"
    args.llm_model_name = "bert"
    args.batch_size = 32
    args.eval_batch_size = 32
    train_loader, val_loader, test_loader = dataset_loader(
        dataset=dataset, args=args, collate_fn=collate_fn
    )
    # Step 3: Build Model
    model = GNNClassifier(args=args, n_classes=n_classes)
    print("Loaded BERT! (frozen)")
    print("Loaded model on device", model.device)

    # Step 4.a Set loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Step 4.b Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": 1e-3, "weight_decay": args.wd},
        ],
        betas=(0.9, 0.95),
    )
    trainable_params, all_param = model.print_trainable_params()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float("inf")
    best_epoch = 0
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss, accum_loss = 0.0, 0.0

        for step, batch in enumerate(train_loader):

            # Use autocast for mixed precision
            y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(
                model.device
            )

            with autocast(device_type="cuda", dtype=torch.float16):
                batch = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(batch)
                loss = criterion(outputs, y_true)
                loss.backward()

            if False:
                adjust_learning_rate(
                    optimizer.param_groups[0],
                    args.lr,
                    step / len(train_loader) + epoch + 2,
                    args,
                )

            epoch_loss += loss.item()
            accum_loss += loss.item()

            if (step + 1) % args.grad_steps == 0:
                # Update model parameters
                optimizer.step()
                optimizer.zero_grad()
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"Lr": lr})
                wandb.log({"Accum Loss": accum_loss})
                progress_bar.set_description(
                    f"Epoch: {epoch}|{args.num_epochs} Loss: {accum_loss:.4f}"
                )
                accum_loss = 0.0

            progress_bar.update(1)

        if epoch == 1:
            save_checkpoint(model, optimizer, epoch, args, is_best=True)
        wandb.log({"Train Loss (Epoch Mean)": epoch_loss / len(train_loader)})

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                # Move the batch to the correct device

                # Use autocast for mixed precision
                # with autocast(device_type="cuda"):
                output = model(batch)
                y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(
                    model.device
                )
                loss = criterion(output, y_true)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            wandb.log({"Val Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f"Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

        torch.cuda.empty_cache()
        gc.collect()

    # Step 5. Evaluating
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    model.eval()
    num_testing_steps = len(test_loader)
    progress_bar_test = tqdm(range(num_testing_steps))
    progress_bar_test.set_description("Evaluation on test set")
    test_accuracies = []

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output: torch.Tensor = model.predict(batch)
            y_true = [class2idx[c] for c in batch["label"]]
            y_true: torch.Tensor = torch.tensor(y_true).to(model.device)
            batch_accuracy = torch.mean((output == y_true).float())
            test_accuracies.append(batch_accuracy.item())
        progress_bar_test.update(1)

    # Step 6. Post-processing & compute metrics
    acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Test Acc {acc}")
    wandb.log({"Test Acc": acc})


if __name__ == "__main__":
    args: Namespace = parse_args_llama()
    main(args=args)
    torch.cuda.empty_cache()
    gc.collect()
