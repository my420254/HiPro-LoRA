#!/usr/bin/env python3

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


MODELS = [
    "roberta-base",
    "hfl/chinese-macbert-base",
]

DATASETS = [
    ("SetFit/sst5", None),
    ("Um1neko/smp2020", None),
    ("tweet_eval", "sentiment"),
]


def main() -> None:
    for model_name in MODELS:
        print(f"[prefetch] model {model_name}", flush=True)
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)

    for dataset_name, config_name in DATASETS:
        label = dataset_name if config_name is None else f"{dataset_name}/{config_name}"
        print(f"[prefetch] dataset {label}", flush=True)
        if config_name is None:
            load_dataset(dataset_name)
        else:
            load_dataset(dataset_name, config_name)


if __name__ == "__main__":
    main()
