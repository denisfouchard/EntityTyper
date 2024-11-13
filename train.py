"""Training script for the GraphLLMClassifier model"""

import os
import wandb
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import Trainer, TrainingArguments

# from torch.nn.utils import clip_grad_norm_
from src.model import llama_model_path
from src.model.graph_llm_classifier import GraphLLMClassifier
from src.dataset.dbpedia import DBPediaDataset
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.dataset.utils.mapping import generate_hierarchical_mapping

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


def main(args):
    scaler = GradScaler()

    gc.collect()
    torch.cuda.empty_cache()
    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(
        project=f"{args.project}",
        name=f"{args.dataset}_{args.model_name}_seed{seed}",
        config=args,
    )

    seed_everything(seed=args.seed)

    dataset = DBPediaDataset()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split["train"]]
    val_dataset = [dataset[i] for i in idx_split["val"]]
    test_dataset = [dataset[i] for i in idx_split["test"]]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
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
    model = GraphLLMClassifier(args=args, n_classes=n_classes)
    print("Loaded model on device", model.device)

    # Step 4.a Set loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Step 4.b Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": args.lr, "weight_decay": args.wd},
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

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.dataset}",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_steps=100,
        logging_steps=100,
        no_cuda=True,
        use_cpu=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
    )

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0.0, 0.0

        for step, batch in enumerate(train_loader):

            # Move the batch to the correct device
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Use autocast for mixed precision
            # y_true = batch_one_hot_encode(batch["label"], n_classes).to(model.device)
            y_true = torch.tensor([class2idx[c] for c in batch["label"]]).to(
                model.device
            )
            batch["label"] = y_true

            inputs = {"samples": batch}
            loss = trainer.training_step(model=model, inputs=inputs)
            # Backward pass
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            accum_loss += loss.item()

            print("Accumulated Loss", accum_loss)
            # Update model parameters
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
            # Clear accumulated gradients
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({"Lr": lr})
            wandb.log({"Accum Loss": accum_loss})
            accum_loss = 0.0

            progress_bar.update(1)

        print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}"
        )
        wandb.log({"Train Loss (Epoch Mean)": epoch_loss / len(train_loader)})

        val_loss = 0.0
        eval_output = []
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
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f"Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    path = f"{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv"
    print(f"path: {path}")
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.cuda.empty_cache()
    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
