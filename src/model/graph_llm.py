import contextlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.graph_encoder import GNN_MODEL_MAPPING
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = "<s>[INST]"
EOS_USER = "[/INST]"
EOS = "</s>"

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print("Loading LLAMA")
        kwargs = {
            "device_map": "auto",
            "max_memory": {
                0: "30GiB",
                1: "15GiB",
                2: "15GiB",
            },
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == "True":
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print("Finish loading LLAMA!")

        self.graph_encoder = GNN_MODEL_MAPPING[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples["graph"]
        graphs = graphs.to(self.model.device)
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

    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(
                BOS, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id)
        ).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = (
                labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            )
            input_ids = (
                descriptions.input_ids[i][: self.max_txt_len]
                + questions.input_ids[i]
                + eos_user_tokens.input_ids
                + label_input_ids
            )
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.model.device)
            )
            inputs_embeds = torch.cat(
                [bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0
            )

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (
                inputs_embeds.shape[0] - len(label_input_ids)
            ) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [
                IGNORE_INDEX
            ] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(
                BOS, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id)
        ).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = (
                descriptions.input_ids[i][: self.max_txt_len]
                + questions.input_ids[i]
                + eos_user_tokens.input_ids
            )
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.model.device)
            )
            inputs_embeds = torch.cat(
                [bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0
            )
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True,  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "id": samples["id"],
            "pred": pred,
            "label": samples["label"],
            "question": samples["question"],
            "desc": samples["desc"],
        }
