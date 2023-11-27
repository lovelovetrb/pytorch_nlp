import torch

# embedding層のパラメータなどを固定できる
    def freeze(model: torch.nn.Module):
        """
        Freezes module's parameters.
        """
        for parameter in model.parameters():
            parameter.requires_grad = False
