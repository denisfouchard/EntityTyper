"""Training script for the Classifier model"""

import os
from argparse import Namespace
import gc
import torch
import pandas as pd
from tqdm import tqdm
from accelerate import infer_auto_device_map, dispatch_model
from torch.amp import autocast
from src.model import llama_model_path
from src.model import CLASSIFIER_MODEL_MAPPING as MODEL_MAP, EntityClassifier
from src.dataset.dbpedia import DBPediaDataset
from src.config import parse_args_llama
from src.utils.checkpoint import save_checkpoint
from src.utils.collate import collate_fn
from src.utils.seeding import seed_everything
from src.utils.lr_scheduling import adjust_learning_rate
from src.dataset.utils.dataloader import dataset_loader
import wandb
from sklearn.metrics import f1_score


def main(args: Namespace) -> None:

    # Step 0: Set up the seed and the device
    seed_everything(seed=args.seed)
    gc.collect()
    torch.cuda.empty_cache()

    # Step 1: Set up wandb
    name = "DBPedia60K - GNN + Frozen LLM"
    wandb.init(
        project=f"{args.project}",
        name=name,
        config=args,
    )

    # Step 2: Load the dataset
    dataset = DBPediaDataset(
        retrieval=args.retrieval,
        version=args.dataset_version,
        entity_desc=args.entity_description,
        summary=False,
    )
    class2idx, idx2class = dataset.class2idx, dataset.idx2class
    n_classes = len(class2idx)
    train_loader, val_loader, test_loader = dataset_loader(
        dataset=dataset, args=args, collate_fn=collate_fn
    )

    # Step 3: Build Model
    entity_classifier: EntityClassifier = MODEL_MAP[args.model_name]
    args.llm_model_path = llama_model_path[args.llm_model_name]
    if args.checkpoint_path != "":
        model = entity_classifier.from_pretrained(
            args=args, n_classes=n_classes, model_path=args.checkpoint_path
        )
    else:
        model = entity_classifier(args=args, n_classes=n_classes)

    accelerate = False
    if accelerate:
        device_map = infer_auto_device_map(
            model, no_split_module_classes=["LlamaDecoderLayer"]
        )
        model = dispatch_model(model, device_map=device_map)
    print("Loaded model on device", model.device)

    # Step 4 : Training Setup
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": args.lr, "weight_decay": args.wd},
        ],
        betas=(0.9, 0.95),
    )
    trainable_params, all_param = model.print_trainable_params()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}" # noqa
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

            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)

            if (step + 1) % args.grad_steps == 0:
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

        val_loss = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                batch = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with autocast(device_type="cuda", dtype=torch.float16):
                    output = model(batch)
                    y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(
                        model.device
                    )
                    loss = criterion(output, y_true)
                    val_loss.append(loss.item())
            val_loss = sum(val_loss) / len(val_loss)
            print(
                f"Epoch {epoch}|{args.num_epochs} Val Loss: {val_loss}, Best Val Loss: {best_val_loss}" # noqa
            )
            wandb.log({"Val Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

        torch.cuda.empty_cache()
        gc.collect()

    # Step 6. Evaluating
    print("==== Evaluating ====")
    os.makedirs(f"{args.output_dir}/{args.dataset}/evaluation/", exist_ok=True)

    path = f"{args.output_dir}/{args.dataset}/evaluation/Results-{name.replace(" ", "")}.csv" # noqa
    print(f"Results saving path: {path}")
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    progress_bar_test.set_description("Evaluation Accuracy on Test Set")
    test_accuracies = []
    true_labels = []
    pred_labels = []
    for _, batch in enumerate(test_loader):
        # Append the list of true labels to the dataframe column
        true_labels.extend([label for label in batch["label"]])
        y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(model.device)
        with torch.no_grad():
            y_pred = model.predict(batch)
            batch_accuracy = torch.mean((y_pred == y_true).float())
            test_accuracies.append(batch_accuracy.item())

        pred_labels.extend([label for label in [idx2class[i] for i in y_pred]])

        progress_bar_test.update(1)

    acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Test Accuracy {acc}")
    wandb.log({"Test Acc": acc})
    # Compute F1 Micro and Macro
    f1_micro = f1_score(true_labels, pred_labels, average="micro")
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    print(f"F1 Micro: {f1_micro}, F1 Macro: {f1_macro}")
    wandb.log({"F1 Micro": f1_micro, "F1 Macro": f1_macro})

    # Save the results to a csv file
    results = pd.DataFrame({"true_labels": true_labels, "pred_labels": pred_labels})
    results.to_csv(path, index=False)


if __name__ == "__main__":
    config_args: Namespace = parse_args_llama()
    main(args=config_args)
    torch.cuda.empty_cache()
    gc.collect()
