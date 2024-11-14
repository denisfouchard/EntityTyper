from argparse import Namespace, ArgumentParser


def csv_list(string):
    return string.split(",")


def parse_args_llama() -> Namespace:
    parser = ArgumentParser(description="G-Retriever")

    parser.add_argument("--model_name", type=str, default="graph_llm_classifier")
    parser.add_argument("--project", type=str, default="DBPediaTyping")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="dbpedia")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_steps", type=int, default=4)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=3)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default="3-8b")
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
        default="/home/infres/dfouchard-21/G-Retriever/output/dbpedia/model_name_graph_llm_classifier_llm_model_name_3-8b_llm_frozen_False_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth",
    )

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default="gt")
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    args = parser.parse_args()
    return args
