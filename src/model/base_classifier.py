import torch


# This is a abstract class that is not used in the codebase
class EntityClassifier(torch.nn.Module):
    def forward(self, samples):
        raise NotImplementedError

    def predict(self, samples) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(samples)
            return torch.argmax(outputs, dim=-1)

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    @staticmethod
    def from_pretrained(args, n_classes, model_path):
        raise NotImplementedError
