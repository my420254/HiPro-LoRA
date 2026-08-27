#!/usr/bin/env python
# coding: utf-8

import argparse
import fcntl
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm


MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen2.5-7B-Instruct",
        "local_path": None,
    },
    "llama": {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "display_name": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "local_path": None,
    },
}

DATASET_CONFIGS = {
    "SMP2020": {
        "labels": {
            0: "anger",
            1: "sadness",
            2: "fear",
            3: "neutral",
            4: "happy",
            5: "surprise",
        },
        "fallback": 3,
        "test_per_class": 50,
    },
    "SST-5": {
        "labels": {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive",
        },
        "fallback": 2,
        "test_per_class": 80,
    },
    "TweetEval": {
        "labels": {
            0: "negative",
            1: "neutral",
            2: "positive",
        },
        "fallback": 1,
        "test_per_class": 50,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_df(dataset_name: str, split: str) -> pd.DataFrame:
    if dataset_name == "SMP2020":
        ds = load_dataset("Um1neko/smp2020", split=split)
        df = pd.DataFrame(ds).rename(columns={"content": "text"})
    elif dataset_name == "SST-5":
        ds = load_dataset("SetFit/sst5", split=split)
        df = pd.DataFrame(ds)
        if "sentence" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"sentence": "text"})
    elif dataset_name == "TweetEval":
        ds = load_dataset("tweet_eval", "sentiment", split=split)
        df = pd.DataFrame(ds)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = df[["text", "label"]].dropna().copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def sample_balanced(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    sampled = []
    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        n = min(len(class_df), per_class)
        sampled.append(class_df.sample(n=n, random_state=seed))
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_examples(dataset_name: str, k_shot: int, seed: int) -> pd.DataFrame:
    if k_shot == 0:
        return pd.DataFrame(columns=["text", "label"])
    train_df = standardize_df(dataset_name, "train")
    return sample_balanced(train_df, k_shot, seed)


def get_test_set(dataset_name: str, seed: int, per_class_override: int | None = None) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[dataset_name]
    test_df = standardize_df(dataset_name, "test")
    per_class = per_class_override if per_class_override is not None else cfg["test_per_class"]
    return sample_balanced(test_df, per_class, seed)


def format_examples(dataset_name: str, examples_df: pd.DataFrame) -> str:
    if examples_df.empty:
        return ""
    lines = []
    if dataset_name == "SMP2020":
        lines.append("参考示例：")
        for _, row in examples_df.sort_values(["label", "text"]).iterrows():
            lines.append(f'文本："{row["text"]}" -> 答案：{int(row["label"])}')
    else:
        lines.append("Examples:")
        for _, row in examples_df.sort_values(["label", "text"]).iterrows():
            lines.append(f'Text: "{row["text"]}" -> Answer: {int(row["label"])}')
    return "\n".join(lines)


def method_name(k_shot: int, prompt_style: str) -> str:
    if k_shot == 0:
        return "Zero-Shot CoT" if prompt_style == "cot" else "Zero-Shot (No CoT)"
    return f"Balanced {k_shot}-Shot CoT" if prompt_style == "cot" else f"Balanced {k_shot}-Shot (No CoT)"


def build_prompt(
    dataset_name: str,
    text: str,
    examples_df: pd.DataFrame,
    k_shot: int,
    prompt_style: str,
) -> tuple[str, str]:
    if prompt_style not in {"no-cot", "cot"}:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    if dataset_name == "SMP2020":
        if prompt_style == "cot":
            task_desc = (
                "任务：判断以下中文文本的情感类别。请先给出简短推理，再输出最终答案。\n"
                "选项：0:愤怒, 1:悲伤, 2:恐惧, 3:中性, 4:高兴, 5:惊奇"
            )
            target = (
                f'请先简短说明判断依据，然后单独一行输出“最终答案：<数字ID>”。\n'
                f'文本："{text}"\n推理过程：'
            )
        else:
            task_desc = (
                "任务：判断以下中文文本的情感类别。\n"
                "选项：0:愤怒, 1:悲伤, 2:恐惧, 3:中性, 4:高兴, 5:惊奇"
            )
            target = f'请仅输出一个数字 ID (0-5)，不要输出任何解释。\n文本："{text}"\n答案：'
    elif dataset_name == "SST-5":
        if prompt_style == "cot":
            task_desc = (
                "Task: Classify the sentiment of the text. Reason briefly, then give the final answer.\n"
                "Options: 0:Very Negative, 1:Negative, 2:Neutral, 3:Positive, 4:Very Positive"
            )
            target = (
                f'Briefly explain the decision, then end with a separate line '
                f'"Final Answer: <numeric ID>".\nText: "{text}"\nReasoning:'
            )
        else:
            task_desc = (
                "Task: Classify the sentiment of the text.\n"
                "Options: 0:Very Negative, 1:Negative, 2:Neutral, 3:Positive, 4:Very Positive"
            )
            target = f'Return ONLY the numeric ID (0-4). Do not explain.\nText: "{text}"\nAnswer:'
    elif dataset_name == "TweetEval":
        if prompt_style == "cot":
            task_desc = (
                "Task: Classify tweet sentiment. Reason briefly, then give the final answer.\n"
                "Options: 0:Negative, 1:Neutral, 2:Positive"
            )
            target = (
                f'Briefly explain the decision, then end with a separate line '
                f'"Final Answer: <numeric ID>".\nText: "{text}"\nReasoning:'
            )
        else:
            task_desc = "Task: Classify tweet sentiment.\nOptions: 0:Negative, 1:Neutral, 2:Positive"
            target = f'Return ONLY the numeric ID (0-2). Do not explain.\nText: "{text}"\nAnswer:'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    method = method_name(k_shot, prompt_style)
    examples = format_examples(dataset_name, examples_df)
    if examples:
        return f"{task_desc}\n{examples}\n\n{target}", method
    return f"{task_desc}\n\n{target}", method


def parse_prediction(response: str, dataset_name: str, prompt_style: str) -> int:
    cfg = DATASET_CONFIGS[dataset_name]
    max_id = max(cfg["labels"])

    if prompt_style == "cot":
        final_answer_matches = re.findall(
            r"(?:Final\s*Answer|最终答案|Answer|答案)\s*[:：]?\s*(\d)",
            response,
            flags=re.IGNORECASE,
        )
        for raw_pred in reversed(final_answer_matches):
            pred = int(raw_pred)
            if 0 <= pred <= max_id:
                return pred

    matches = re.findall(r"\d", response)
    if matches:
        candidates = reversed(matches) if prompt_style == "cot" else matches
        for raw_pred in candidates:
            pred = int(raw_pred)
            if 0 <= pred <= max_id:
                return pred
    return cfg["fallback"]


def resolve_model_source(model_key: str, model_path: str | None = None) -> str:
    cfg = MODEL_CONFIGS[model_key]
    if model_path:
        return model_path
    local_path = cfg.get("local_path")
    if local_path and Path(local_path).exists():
        return local_path
    return cfg["model_id"]


def load_model(model_key: str, model_path: str | None = None, local_files_only: bool = False):
    cfg = MODEL_CONFIGS[model_key]
    model_source = resolve_model_source(model_key, model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading {cfg['display_name']} from {model_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    return tokenizer, model, params_m


def run_one(
    tokenizer,
    model,
    params_m: float,
    model_key: str,
    dataset_name: str,
    k_shot: int,
    seed: int,
    output_dir: Path,
    prompt_style: str,
    max_new_tokens_no_cot: int,
    max_new_tokens_cot: int,
    test_per_class_override: int | None = None,
):
    examples_df = sample_examples(dataset_name, k_shot, seed)
    test_df = get_test_set(dataset_name, seed, test_per_class_override)
    prompt_template, method = build_prompt(dataset_name, "[TEXT_TO_CLASSIFY]", examples_df, k_shot, prompt_style)

    preds = []
    responses = []
    labels = test_df["label"].tolist()
    if prompt_style == "cot":
        sys_msg = (
            "你是严谨的情感分析专家。请先给出简短推理，再输出最终答案。"
            if dataset_name == "SMP2020"
            else "You are a logical sentiment analysis expert. Reason briefly, then provide the final answer."
        )
        max_new_tokens = max_new_tokens_cot
    else:
        sys_msg = "You are a sentiment classification system. You must output ONLY a single digit as the answer."
        max_new_tokens = max_new_tokens_no_cot

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for text in tqdm(test_df["text"], desc=f"{MODEL_CONFIGS[model_key]['display_name']} | {dataset_name} | {method}"):
        content, _ = build_prompt(dataset_name, text, examples_df, k_shot, prompt_style)
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        responses.append(response)
        preds.append(parse_prediction(response, dataset_name, prompt_style))

    elapsed = time.time() - start
    latency_ms = (elapsed / len(test_df)) * 1000
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)

    display_name = MODEL_CONFIGS[model_key]["display_name"]
    safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
    safe_model = display_name.replace("/", "_")
    cases_path = output_dir / "cases" / f"{safe_model}_{dataset_name}_{safe_method}_cases.csv"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": test_df["text"],
            "true_label": labels,
            "pred_label": preds,
            "llm_response": responses,
            "k_shot": k_shot,
            "prompt_style": prompt_style,
            "protocol": "balanced-held-out-test",
        }
    ).to_csv(cases_path, index=False)

    examples_path = output_dir / "prompts" / f"{dataset_name}_{k_shot}shot_examples.csv"
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples_path.exists():
        examples_df.to_csv(examples_path, index=False)
    prompt_template_path = output_dir / "prompts" / f"{dataset_name}_{k_shot}shot_{prompt_style}_prompt_template.txt"
    prompt_template_path.write_text(prompt_template, encoding="utf-8")

    return {
        "Dataset": dataset_name,
        "Model": display_name,
        "Method": method,
        "Macro-F1": macro_f1,
        "Accuracy": accuracy,
        "Inference_Time_ms": latency_ms,
        "Peak_Memory_MB": peak_memory_mb,
        "Params_M": params_m,
        "Protocol": "balanced-held-out-test",
    }


def append_result(output_dir: Path, row: dict) -> None:
    result_path = output_dir / "llm_fewshot_results.csv"
    lock_path = output_dir / "llm_fewshot_results.csv.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        df_new = pd.DataFrame([row])
        if result_path.exists():
            df_old = pd.read_csv(result_path)
            key_cols = ["Dataset", "Model", "Method", "Protocol"]
            mask = pd.Series([True] * len(df_old))
            for col in key_cols:
                mask &= df_old[col].astype(str) == str(row[col])
            df_old = df_old[~mask]
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(result_path, index=False)
        fcntl.flock(lock_fh, fcntl.LOCK_UN)


def has_existing_result(output_dir: Path, dataset_name: str, model_key: str, method: str) -> bool:
    result_path = output_dir / "llm_fewshot_results.csv"
    if not result_path.exists():
        return False
    df = pd.read_csv(result_path)
    display_name = MODEL_CONFIGS[model_key]["display_name"]
    mask = (
        (df["Dataset"].astype(str) == dataset_name)
        & (df["Model"].astype(str) == display_name)
        & (df["Method"].astype(str) == method)
        & (df["Protocol"].astype(str) == "balanced-held-out-test")
    )
    return bool(mask.any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--datasets", default="SMP2020,SST-5,TweetEval")
    parser.add_argument("--shots", default="5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "few_shot_results"))
    parser.add_argument("--prompt-style", choices=["no-cot", "cot", "both"], default="no-cot")
    parser.add_argument("--max-new-tokens-no-cot", type=int, default=5)
    parser.add_argument("--max-new-tokens-cot", type=int, default=150)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--test-per-class-override", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    prompt_styles = ["no-cot", "cot"] if args.prompt_style == "both" else [args.prompt_style]

    tokenizer, model, params_m = load_model(args.model, args.model_path, args.local_files_only)
    for k_shot in shots:
        for dataset_name in datasets:
            for prompt_style in prompt_styles:
                method = method_name(k_shot, prompt_style)
                if args.skip_existing and has_existing_result(output_dir, dataset_name, args.model, method):
                    print(
                        f"Skipping existing result: model={args.model} dataset={dataset_name} method={method}",
                        flush=True,
                    )
                    continue
                row = run_one(
                    tokenizer,
                    model,
                    params_m,
                    args.model,
                    dataset_name,
                    k_shot,
                    args.seed,
                    output_dir,
                    prompt_style,
                    args.max_new_tokens_no_cot,
                    args.max_new_tokens_cot,
                    args.test_per_class_override,
                )
                append_result(output_dir, row)
                print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
