from argparse import Namespace, ArgumentParser


def csv_list(string):
    return string.split(",")


def parse_args_llama() -> Namespace:
    parser = ArgumentParser(description="G-Retriever")

    parser.add_argument("--model_name", type=str, default="graph_llm")
    parser.add_argument("--project", type=str, default="DBPediaTyping")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="dbpedia")
    parser.add_argument("--dataset_version", type=str, default="60k")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=3)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=10)

    # Retrieval related
    parser.add_argument("--retrieval", type=bool, default=False)

    # Description related
    parser.add_argument("--entity_description", type=bool, default=False)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default="llama")
    parser.add_argument("--llm_model_path", type=str, default="")
    parser.add_argument("--llm_frozen", type=str, default="False")
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_memory", type=csv_list, default=[24])
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
    )

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default="gt")
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.2)

    args = parser.parse_args()
    return args
