import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torchmetrics.classification import MultilabelF1Score
from transformers import AutoModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


class DebertaGenresModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 20,
        lr_backbone: float = 1e-5,
        lr_head: float = 5e-5,
        weight_decay: float = 0.01,
        threshold: float = 0.5,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.backbone.train()

        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden, num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_f1 = MultilabelF1Score(num_labels=num_labels, average="micro", threshold=threshold)
        self.val_f1 = MultilabelF1Score(num_labels=num_labels, average="micro", threshold=threshold)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]
        cls_vec = self.dropout(cls_vec)
        logits = self.classifier(cls_vec)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)

        probs = torch.sigmoid(logits)
        self.train_f1.update(probs, labels.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_micro_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)

        probs = torch.sigmoid(logits)
        self.val_f1.update(probs, labels.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_micro_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.classifier.parameters(), "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer

