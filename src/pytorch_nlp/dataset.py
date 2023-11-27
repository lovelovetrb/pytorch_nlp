import json

import torch
from torch.utils.data import Dataset
from transformers import BertJapaneseTokenizer


class baseDataset(Dataset):
    def __init__(
        self, data_path: str, tokenizer: BertJapaneseTokenizer, model_max_length: int
    ) -> None:
        super().__init__()
        # json ファイルを読み込む
        self.data = json.load(open(data_path, "r"))
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id

    # ここで取り出すデータを指定している
    def __getitem__(self, index: int):
        # トークン化
        tokenized_sentence = self.tokenize_chunk(self.data[index]["sentence"])

        return {
            "input_ids": tokenized_sentence["input_ids"].squeeze(0),
            "attention_mask": tokenized_sentence["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.data[index]["labels"], dtype=torch.long),
            "sentence": self.data[index]["sentence"],
        }

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_chunk(self, sentence: str):
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )
        return tokenized_sentence
