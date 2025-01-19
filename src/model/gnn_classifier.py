import torch
import torch.nn as nn
from src.model.base_classifier import EntityClassifier
from src.model.graph_encoder import GNN_MODEL_MAPPING, GraphEncoder
from src.utils.lm_modeling import load_sbert_to_device
from contextlib import contextmanager


@contextmanager
def conditional_no_grad(frozen: bool):
    if frozen:
        with torch.no_grad():
            yield
    else:
        yield


IGNORE_INDEX = -100


class CustomClassificationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, samples) -> torch.Tensor:
        # Take the last hidden state
        return self.mlp(samples)


class GNNClassifier(EntityClassifier):

    def __init__(self, args, n_classes: int, **kwargs):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.n_classes = n_classes
        self.entity_description = args.entity_description
        self.freezing_bert = args.llm_frozen

        # Load SBERT
        self._sbert_model, self._sbert_tokenizer = load_sbert_to_device(device)

        self.graph_encoder: GraphEncoder = GNN_MODEL_MAPPING[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(device)

        self.classifier: CustomClassificationMLP = CustomClassificationMLP(
            input_dim=args.gnn_hidden_dim + 1024,  # 1024 is the SBERT embedding size
            hidden_dim=1024,
            num_classes=n_classes,
        ).to(device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def encode_entities(self, samples: str) -> torch.Tensor:

        entities_embeddings = []
        if self.entity_description and len(samples["desc"]) > 0:
            entities = samples["desc"]
        else:
            entities = samples["entity"]
        with conditional_no_grad(self.freezing_bert):
            for entity in entities:
                entity_encoding = self._sbert_tokenizer(
                    text=entity, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                entity_embedding = self._sbert_model.forward(
                    input_ids=entity_encoding.input_ids,
                    att_mask=entity_encoding.attention_mask,
                ).to(self.device)
                entities_embeddings.append(entity_embedding)
        entities_embeddings = torch.stack(entities_embeddings)
        return entities_embeddings

    def encode_graphs(self, samples) -> torch.Tensor:
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
        # graph_embeds = graph_embeds.unsqueeze(0)
        entity_embeds = self.encode_entities(samples)
        entity_embeds = entity_embeds.squeeze(1)

        # Concatenate the embeddings
        graph_embeds = torch.cat((graph_embeds, entity_embeds), dim=1)

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
