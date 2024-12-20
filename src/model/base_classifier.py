import torch


# This is a abstract class that is not used in the codebase
class BaseClassifier(torch.nn.Module):
    def forward(self, samples):
        raise NotImplementedError

    def predict(self, samples):
        raise NotImplementedError

    def print_trainable_params(self):
        raise NotImplementedError

    @staticmethod
    def from_pretrained(args, n_classes, model_path):
        raise NotImplementedError
