import torch
import torch.nn as nn
from transformers import (
    LlamaModel,
    AutoTokenizer,
)
from src.model.base_classifier import BaseClassifier
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = "<s>[INST]"
EOS_USER = "[/INST]"
EOS = "</s>"

IGNORE_INDEX = -100


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden_states) -> torch.Tensor:
        # Take the last hidden state
        last_hidden_state = hidden_states[:, -1, :]
        return self.mlp(last_hidden_state)


class LLMClassifier(BaseClassifier):
    """No Graph encoding. Only LLM model for classification"""

    def __init__(self, args, n_classes: int, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print("Loading LLAMA")
        kwargs = {
            "device_map": "auto",
            "max_memory": {
                0: "25GiB",
            },
            "revision": "main",
        }

        self.n_classes = n_classes

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        language_model: LlamaModel = LlamaModel.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            **kwargs,
        )

        # Remove the original final layer
        language_model.config.is_decoder = (
            False  # Disable decoding since it's not needed for classification
        )

        if args.llm_frozen == "True":
            print("Freezing LLAMA!")
            for name, param in language_model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            language_model = prepare_model_for_kbit_training(language_model)
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
                task_type="classification",
            )
            language_model = get_peft_model(language_model, config)

        self.language_model = language_model
        print("Finish loading LLAMA!")

        self.classifier: ClassificationHead = ClassificationHead(
            input_dim=4096,
            hidden_dim=1024,
            num_classes=n_classes,
        ).to(self.language_model.device)

        self.word_embedding = self.language_model.get_input_embeddings()

        print(
            "Classifier trainable parameters:",
            sum(p.numel() for p in self.classifier.parameters()),
        )

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, samples) -> torch.Tensor:
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        # eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .to(self.language_model.device)
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(self.language_model.device)
        ).unsqueeze(0)

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
                torch.tensor(input_ids).to(self.language_model.device)
            )
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

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

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(
            self.language_model.device
        )
        attention_mask = torch.tensor(batch_attention_mask).to(
            self.language_model.device
        )

        hidden_states = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        ).last_hidden_state

        output = self.classifier.forward(hidden_states)

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
    def from_pretrained(args, n_classes, model_path) -> "LLMClassifier":
        print(f"Loading model from {model_path}")
        trained_llm_classifier = LLMClassifier(args, n_classes)
        checkpoint = torch.load(model_path)
        trained_llm_classifier.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded model from checkpoint!")
        return trained_llm_classifier


if __name__ == "__main__":
    from src.config import parse_args_llama
    from src.model import llama_model_path

    args = parse_args_llama()
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = LLMClassifier(args, 2)
    print(model)
    trainable_params, all_param = model.print_trainable_params()
    print(f"Trainable params: {trainable_params}, All params: {all_param}")
    print("Done!")
    print(model.device)
