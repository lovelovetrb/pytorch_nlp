import torch
import torch.nn as nn
from transformers import BertModel


class BaseModel(nn.Module):
    def __init__(self, model_name: str, dropout=0.2) -> None:
        super(BaseModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        output = self.dropout(output)
        output = self.classifier(output[0])
        output = self.softmax(output)

        return output
