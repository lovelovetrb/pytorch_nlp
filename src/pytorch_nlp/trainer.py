import os

import torch
# import wandb
from tqdm import tqdm

from src.pytorch_nlp.dataset import baseDataset
from src.pytorch_nlp.model import BaseModel
from src.pytorch_nlp.util import logger


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        train_dataset: baseDataset,
        config: dict,
        args: dict,
    ) -> None:
        logger.info("Initializing trainer...")
        self.config = config
        self.args = args

        self.model = model
        # wandb.watch(self.model)

        self.train_dataset = train_dataset

        # AMP
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["train"]["lr"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        # Data sampling
        sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["train"]["batch_size"],
            sampler=sampler,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

        torch.backends.cudnn.benchmark = True

    def train(self) -> None:
        logger.info("Start training...")
        for i in range(self.config["train"]["epoch"]):
            logger.info(f"Epoch: {i + 1}")
            iter_bar = tqdm(self.train_dataloader, disable=self.args.is_master)
            for batch_label, batch_data in iter_bar:
                self.step(batch_label=batch_label, batch_data=batch_data)
            self.end_epoch()

    def step(self, batch_label, batch_data) -> None:
        self.optimizer.zero_grad()
        batch_label = batch_label.to(self.args.device)  # [batch_size, seq_len, seq_len]

        with torch.cuda.amp.autocast():
            output = self.model(
                # [batch_size, 1, seq_len] -> [batch_size, seq_len]
                input_ids=batch_data.input_ids.to(self.args.device),
                attention_mask=batch_data.attention_mask.to(self.args.device),
            )

            assert batch_label.shape == output.shape
            loss = self.loss_fn(output, batch_label)

        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.accuracy = (output.argmax(dim=1) == batch_label).sum().item() / len(
            batch_label
        )
        logger.info(f"accuracy: {self.accuracy}")
        logger.info(f"loss: {loss}")

        # wandb.log({"accuracy": self.accuracy, "loss": loss})

    def end_epoch(self) -> None:
        if not self.args.is_master:
            return
        self.save_model()

    def save_model(self) -> None:
        torch.save(
            self.model,
            self.config["train"]["save_path"],
        )
