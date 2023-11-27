import argparse
import os

import torch
import yaml
from transformers import BertJapaneseTokenizer

from src.pytorch_nlp.dataset import baseDataset
from src.pytorch_nlp.model import BaseModel
from src.pytorch_nlp.trainer import Trainer
from src.pytorch_nlp.util import init_config, init_gpu, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "visible_gpu",
        type=int,
        nargs="*",
        default=[],
        help="visible gpu",
        required=True,
    )
    args = parser.parse_args()

    # config.yamlの読み込み
    logger.info("Loading config.yaml...")
    with open("src/dependency_llm/config.yaml") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.visible_gpu])

    init_gpu(args)
    init_config(config)

    tokenizer = BertJapaneseTokenizer.from_pretrained(config["basic"]["model_name"])
    args.model_max_length = tokenizer.max_model_input_sizes[
        config["basic"]["model_name"]
    ]

    model = BaseModel(config["basic"]["model_name"])
    model.to("cuda")
    model.train()

    dataset = baseDataset(
        data_path=config["basic"]["data_path"],
        model_max_length=args.model_max_length,
        tokenizer=tokenizer,
    )

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainer = Trainer(
        model=model, train_dataset=train_dataset, config=config, args=args
    )
    trainer.train()
    logger.info("Finish training!")


if __name__ == "__main__":
    main()
