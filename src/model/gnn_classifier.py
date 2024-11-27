import torch
import torch.nn as nn
from src.model.graph_encoder import GNN_MODEL_MAPPING, GraphEncoder

IGNORE_INDEX = -100


class CustomClassificationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden_states) -> torch.Tensor:
        # Take the last hidden state
        last_hidden_state = hidden_states[:, -1, :]
        return self.mlp(last_hidden_state)


class GNNClassifier(nn.Module):

    def __init__(self, args, n_classes: int, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.n_classes = n_classes

        self.graph_encoder: GraphEncoder = GNN_MODEL_MAPPING[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.device)

        self.classifier: CustomClassificationMLP = CustomClassificationMLP(
            input_dim=args.gnn_hidden_dim,
            hidden_dim=1024,
            num_classes=n_classes,
        ).to(self.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def encode_graphs(self, samples):
        graphs = samples["graph"]
        graphs = graphs.to(self.device)
        n_embeds: torch.Tensor = self.graph_encoder(
            graphs.x, graphs.edge_index.long(), graphs.edge_attr
        )[0]

        # mean pooling using vanilla PyTorch
        batch_size = graphs.batch.max().item() + 1
        g_embeds_sum = torch.zeros(batch_size, n_embeds.size(1), device=n_embeds.device)
        g_embeds_count = torch.zeros(batch_size, device=n_embeds.device)

        g_embeds_sum.scatter_add_(
            0, graphs.batch.unsqueeze(-1).expand_as(n_embeds), n_embeds
        )
        g_embeds_count.scatter_add_(
            0,
            graphs.batch,
            torch.ones_like(graphs.batch, device=n_embeds.device, dtype=torch.float),
        )

        g_embeds = g_embeds_sum / g_embeds_count.unsqueeze(-1)

        return g_embeds

    def forward(self, samples) -> torch.Tensor:
        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        # Classification
        output = self.classifier.forward(graph_embeds)

        return output

    def predict(self, samples) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(samples)
            return torch.argmax(outputs, dim=-1)

    def print_trainable_params(self) -> tuple[int, int]:
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    @staticmethod
    def from_pretrained(args, n_classes, model_path) -> "GNNClassifier":
        print(f"Loading model from {model_path}")
        trained_graph_classifier: GNNClassifier = GNNClassifier(args, n_classes)
        checkpoint = torch.load(model_path)
        trained_graph_classifier.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded model from checkpoint!")
        return trained_graph_classifier


if __name__ == "__main__":
    from src.config import parse_args_llama

    args = parse_args_llama()
    model = GNNClassifier(args, 2)
    print(model)
    trainable_params, all_param = model.print_trainable_params()
    print(f"Trainable params: {trainable_params}, All params: {all_param}")
    print("Done!")
    print(model.device)
