import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from model_module import DebertaGenresModule


CKPT_PATH = "checkpoints/deberta-epoch=07-val_micro_f1=0.6672.ckpt"
MODEL_NAME = "microsoft/deberta-v3-base"
OUT_DIR = Path("artifacts")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv("data.csv")
    genres = df.columns[2:].tolist()

    with open(OUT_DIR / "genres.json", "w", encoding="utf-8") as f:
        json.dump(genres, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": MODEL_NAME, "num_labels": len(genres)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    (OUT_DIR / "tokenizer").mkdir(exist_ok=True)
    tokenizer.save_pretrained(OUT_DIR / "tokenizer")

    model = DebertaGenresModule.load_from_checkpoint(
        CKPT_PATH,
        model_name=MODEL_NAME,
        num_labels=len(genres),
        map_location="cpu",
    )
    model.eval()

    (OUT_DIR / "backbone").mkdir(exist_ok=True)
    model.backbone.save_pretrained(OUT_DIR / "backbone")

    torch.save(model.classifier.state_dict(), OUT_DIR / "classifier.pt")


if __name__ == "__main__":
    main()