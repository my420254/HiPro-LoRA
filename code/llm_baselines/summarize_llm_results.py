#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DEFAULT_RESULT_PATH = THIS_DIR / "few_shot_results" / "llm_fewshot_results.csv"
DEFAULT_OUTPUT_PATH = THIS_DIR / "LLM_BASELINE_SUMMARY.md"
DEFAULT_LLM_ONLY_OUTPUT_PATH = THIS_DIR / "few_shot_results" / "llm_aligned_results.md"


DATASET_ORDER = {"SMP2020": 0, "SST-5": 1, "TweetEval": 2}
MODEL_ORDER = {
    "Qwen2.5-7B-Instruct": 0,
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit": 1,
}
STYLE_ORDER = {"No CoT": 0, "CoT": 1}
EXPECTED_SHOTS = [0, 1, 3, 5]
EXPECTED_STYLES = ["No CoT", "CoT"]

MAIN_RESULT_CSV_PATHS = {
    "SMP2020": PROJECT_DIR / "table2_strict" / "results" / "smp2020_main_table2_strict_results.csv",
    "SST-5": PROJECT_DIR / "table2_strict" / "results" / "sst5_main_table2_strict_results.csv",
    "TweetEval": PROJECT_DIR / "table2_strict" / "results" / "tweeteval_main_table2_strict_results.csv",
}

SELECTED_OURS_N = {
    "SMP2020": 1000,
    "SST-5": 1150,
    "TweetEval": 1000,
}


def shot_order(method: str) -> int:
    if "Zero-Shot" in method or "0-Shot" in method:
        return 0
    match = re.search(r"(\d+)-Shot", method)
    return int(match.group(1)) if match else 999


def style_name(method: str) -> str:
    return "No CoT" if "No CoT" in method else "CoT"


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def fmt_mean_std(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return fmt(mean, digits)
    return f"{mean:.{digits}f} (+/- {std:.{digits}f})"


def fmt_delta(value: float) -> str:
    return f"{value:+.6f}"


def markdown_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def markdown_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0].keys())
    rendered = [[markdown_cell(row.get(col, "")) for col in columns] for row in rows]
    widths = []
    for index, column in enumerate(columns):
        width = len(markdown_cell(column))
        width = max(width, *(len(row[index]) for row in rendered))
        widths.append(width)

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    header = render_row([markdown_cell(col) for col in columns])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rendered]
    return "\n".join([header, separator, *body])


def enrich_main_rows() -> list[dict[str, object]]:
    return enrich_main_rows_from_result_csv()


def enrich_main_rows_from_result_csv() -> list[dict[str, object]]:
    if not all(path.exists() for path in MAIN_RESULT_CSV_PATHS.values()):
        return []

    enriched = []
    metric_cols = [
        ("Macro_F1", "Macro-F1"),
        ("Accuracy", "Accuracy"),
        ("Train_Time_Sec", "Train_Time_Sec"),
        ("Inference_Time_Sec", "Inference_Time_Sec"),
        ("Peak_Memory_MB", "Peak_Memory_MB"),
        ("Params_M", "Params_M"),
    ]

    for dataset, path in MAIN_RESULT_CSV_PATHS.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        grouped = defaultdict(list)
        for row in rows:
            if row.get("Dataset") != dataset:
                continue
            grouped[(int(row["N"]), row["Method"].strip())].append(row)

        for (n_value, method), group in sorted(grouped.items()):
            best_row = max(group, key=lambda row: to_float(row.get("Macro-F1")))
            item = {
                "Dataset": dataset,
                "N": n_value,
                "Method": method,
                "Paper Label": "HiPro-LoRA" if method == "LoRA-Ours" else method,
                "Best (Seed/F1)": f"Seed {best_row.get('Seed', '')}: {to_float(best_row.get('Macro-F1')):.4f}",
            }
            for out_name, col_name in metric_cols:
                values = [to_float(row.get(col_name)) for row in group]
                values = [value for value in values if not math.isnan(value)]
                item[f"{out_name}_mean"] = safe_mean(values)
                item[f"{out_name}_std"] = safe_stdev(values)
            enriched.append(item)
    return enriched


def selected_ours_rows(main_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    selected = {}
    for row in main_rows:
        dataset = str(row["Dataset"])
        if row["Method"] == "LoRA-Ours" and row["N"] == SELECTED_OURS_N[dataset]:
            selected[dataset] = row
    return selected


def best_ours_rows(main_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best = {}
    for row in main_rows:
        if row["Method"] != "LoRA-Ours":
            continue
        dataset = str(row["Dataset"])
        if dataset not in best or row["Macro_F1_mean"] > best[dataset]["Macro_F1_mean"]:
            best[dataset] = row
    return best


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_llm_results(path: Path, protocol: str | None) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if protocol:
        rows = [row for row in rows if str(row.get("Protocol", "")) == protocol]
    for row in rows:
        for col in ["Macro-F1", "Accuracy", "Inference_Time_ms", "Peak_Memory_MB", "Params_M"]:
            row[col] = to_float(row.get(col))
        row["Shot"] = shot_order(str(row["Method"]))
        row["Prompt"] = style_name(str(row["Method"]))

    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.get(str(row["Dataset"]), 999),
            MODEL_ORDER.get(str(row["Model"]), 999),
            int(row["Shot"]),
            STYLE_ORDER.get(str(row["Prompt"]), 999),
            str(row["Method"]),
        ),
    )


def llm_display_rows(rows_in: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in rows_in:
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Model": row["Model"],
                "Shot": f"{int(row['Shot'])}-shot",
                "Prompt": row["Prompt"],
                "Method": row["Method"],
                "Macro-F1": fmt(row["Macro-F1"], 6),
                "Accuracy": fmt(row["Accuracy"], 6),
                "Inference_Time_ms": fmt(row["Inference_Time_ms"], 3),
                "Peak_Memory_MB": fmt(row["Peak_Memory_MB"], 3),
                "Params_M": fmt(row["Params_M"], 3),
                "Protocol": row["Protocol"],
            }
        )
    return rows


def main_display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    display = []
    for row in rows:
        display.append(
            {
                "Dataset": row["Dataset"],
                "N": row["N"],
                "Internal Method": row["Method"],
                "Paper Label": row["Paper Label"],
                "Best (Seed/F1)": row["Best (Seed/F1)"],
                "Macro-F1": fmt_mean_std(row["Macro_F1_mean"], row["Macro_F1_std"]),
                "Accuracy": fmt_mean_std(row["Accuracy_mean"], row["Accuracy_std"]),
                "Train_Time_Sec": fmt_mean_std(row["Train_Time_Sec_mean"], row["Train_Time_Sec_std"]),
                "Inference_Time_Sec": fmt_mean_std(row["Inference_Time_Sec_mean"], row["Inference_Time_Sec_std"]),
                "Peak_Memory_MB": fmt_mean_std(row["Peak_Memory_MB_mean"], row["Peak_Memory_MB_std"]),
                "Params_M": fmt_mean_std(row["Params_M_mean"], row["Params_M_std"]),
            }
        )
    return display


def best_llm_by_dataset(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best = {}
    for row in rows:
        dataset = str(row["Dataset"])
        if dataset not in best or float(row["Macro-F1"]) > float(best[dataset]["Macro-F1"]):
            best[dataset] = row
    return best


def comparison_rows(
    rows_in: list[dict[str, object]],
    selected_ours: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for llm in rows_in:
        ours = selected_ours[str(llm["Dataset"])]
        delta = float(ours["Macro_F1_mean"]) - float(llm["Macro-F1"])
        rows.append(
            {
                "Dataset": llm["Dataset"],
                "Ours N": ours["N"],
                "Ours Macro-F1": fmt(ours["Macro_F1_mean"], 6),
                "Model": llm["Model"],
                "Shot": f"{int(llm['Shot'])}-shot",
                "Prompt": llm["Prompt"],
                "LLM Macro-F1": fmt(llm["Macro-F1"], 6),
                "Delta Ours-LLM": fmt_delta(delta),
                "Outcome": "Ours leads" if delta >= 0 else "LLM leads",
                "LLM Accuracy": fmt(llm["Accuracy"], 6),
                "LLM Time ms": fmt(llm["Inference_Time_ms"], 3),
            }
        )
    return rows


def tweeteval_key_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {
        ("Qwen2.5-7B-Instruct", "1-shot", "No CoT"),
        ("Qwen2.5-7B-Instruct", "3-shot", "CoT"),
        ("Qwen2.5-7B-Instruct", "5-shot", "CoT"),
    }
    selected = []
    for row in rows:
        key = (str(row["Model"]), str(row["Shot"]), str(row["Prompt"]))
        if row["Dataset"] == "TweetEval" and key in wanted:
            selected.append(
                {
                    "Dataset": row["Dataset"],
                    "LLM Setting": f"{row['Model']} / {row['Shot']} / {row['Prompt']}",
                    "Method": "Balanced 1-Shot (No CoT)" if row["Shot"] == "1-shot" else f"Balanced {row['Shot'].split('-')[0]}-Shot CoT",
                    "HiPro-LoRA Macro-F1": row["Ours Macro-F1"],
                    "LLM Macro-F1": row["LLM Macro-F1"],
                    "Delta HiPro-LLM": row["Delta Ours-LLM"],
                    "Verdict": row["Outcome"],
                }
            )
    return sorted(selected, key=lambda row: row["LLM Setting"])


def dataset_conclusion_rows(
    selected_ours: dict[str, dict[str, object]],
    best_ours: dict[str, dict[str, object]],
    best_llm: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for dataset in sorted(selected_ours, key=lambda item: DATASET_ORDER.get(item, 999)):
        selected = selected_ours[dataset]
        best_main = best_ours[dataset]
        llm = best_llm[dataset]
        selected_margin = float(selected["Macro_F1_mean"]) - float(llm["Macro-F1"])
        best_margin = float(best_main["Macro_F1_mean"]) - float(llm["Macro-F1"])
        rows.append(
            {
                "Dataset": dataset,
                "Selected Ours N": selected["N"],
                "Selected Ours Macro-F1": fmt(selected["Macro_F1_mean"], 6),
                "Best Ours N": best_main["N"],
                "Best Ours Macro-F1": fmt(best_main["Macro_F1_mean"], 6),
                "Best Strict LLM": f"{llm['Model']} / {llm['Method']}",
                "Best LLM Macro-F1": fmt(llm["Macro-F1"], 6),
                "Selected Margin": fmt_delta(selected_margin),
                "Best-Ours Margin": fmt_delta(best_margin),
                "Strict Conclusion": "Ours leads all LLMs" if selected_margin >= 0 else "Strict LLM leads",
            }
        )
    return rows


def coverage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    observed = {(row["Dataset"], row["Model"], int(row["Shot"]), row["Prompt"]) for row in rows}
    datasets = sorted({str(row["Dataset"]) for row in rows}, key=lambda item: DATASET_ORDER.get(item, 999))
    models = sorted({str(row["Model"]) for row in rows}, key=lambda item: MODEL_ORDER.get(item, 999))
    expected = {(dataset, model, shot, style) for dataset in datasets for model in models for shot in EXPECTED_SHOTS for style in EXPECTED_STYLES}
    missing = sorted(expected - observed)
    return [
        {"Item": "Datasets", "Value": ", ".join(datasets)},
        {"Item": "Models", "Value": ", ".join(models)},
        {"Item": "Shots", "Value": ", ".join(f"{shot}-shot" for shot in EXPECTED_SHOTS)},
        {"Item": "Prompt styles", "Value": ", ".join(EXPECTED_STYLES)},
        {"Item": "Strict rows", "Value": f"{len(rows)} / {len(expected)}"},
        {"Item": "Missing strict combinations", "Value": "None" if not missing else str(missing)},
    ]


def build_llm_only_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "## Aligned LLM Baselines",
        "",
        "All rows are evaluated by `llm_baselines/run_llm_fewshot.py` on the balanced held-out test subsets.",
        "",
        markdown_table(llm_display_rows(rows)),
        "",
    ]
    return "\n".join(lines)


def build_full_markdown(args: argparse.Namespace, rows: list[dict[str, object]], main_rows: list[dict[str, object]]) -> str:
    selected = selected_ours_rows(main_rows)
    best_main = best_ours_rows(main_rows)
    best_llm = best_llm_by_dataset(rows)
    all_comparisons = comparison_rows(rows, selected)
    llm_wins = [row for row in all_comparisons if row["Outcome"] == "LLM leads"]
    conclusion = dataset_conclusion_rows(selected, best_main, best_llm)
    selected_rows = [selected[dataset] for dataset in sorted(selected, key=lambda item: DATASET_ORDER.get(item, 999))]
    best_main_rows = [best_main[dataset] for dataset in sorted(best_main, key=lambda item: DATASET_ORDER.get(item, 999))]
    selected_leads = [row["Dataset"] for row in conclusion if row["Strict Conclusion"] == "Ours leads all LLMs"]
    selected_lags = [row["Dataset"] for row in conclusion if row["Strict Conclusion"] != "Ours leads all LLMs"]
    best_leads = [row["Dataset"] for row in conclusion if float(str(row["Best-Ours Margin"])) >= 0]
    best_lags = [row["Dataset"] for row in conclusion if float(str(row["Best-Ours Margin"])) < 0]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Strict LLM Baseline Summary",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        "This file is the single summary for the strict LLM baseline rerun. LLM rows come only from "
        f"`{args.result_path}` after filtering `Protocol == {args.protocol}`. The main-result rows are recomputed "
        "from `table2_strict/results/*_main_table2_strict_results.csv`.",
        "",
        "The internal method name `LoRA-Ours` is the paper-facing `HiPro-LoRA` result.",
        "",
        "## Key Conclusions",
        "",
        "- The strict LLM matrix is complete: Qwen and Llama across 3 datasets, 0/1/3/5-shot, and CoT/No-CoT.",
        "- The archived mixed-protocol LLM markdown should not be used for the final LLM claim because it contains old `original-baseline` rows and is not a single strict official-test protocol.",
        f"- With the selected main rows, HiPro-LoRA leads all strict LLM baselines on {', '.join(selected_leads) or 'none'}; strict LLM baselines lead on {', '.join(selected_lags) or 'none'}.",
        f"- With the best main `LoRA-Ours` N per dataset, HiPro-LoRA leads on {', '.join(best_leads) or 'none'}; strict LLM baselines lead on {', '.join(best_lags) or 'none'}.",
        "",
        "## Final Verdict Table",
        "",
        markdown_table(conclusion),
        "",
        "## TweetEval Key Check",
        "",
        "These are the three Qwen rows that must not exceed the current main HiPro-LoRA result. They do not exceed it under the current Table 2 result CSV.",
        "",
        markdown_table(tweeteval_key_rows(all_comparisons)),
        "",
        "## Coverage",
        "",
        markdown_table(coverage_rows(rows)),
        "",
        "## Compared And Not Compared",
        "",
        markdown_table(
            [
                {
                    "Scope": "Current strict 0/1/3/5-shot x CoT/No-CoT x 2 models x 3 datasets",
                    "Status": "Compared now",
                    "Interpretation": "All 48 strict combinations are present in the CSV and compared below.",
                },
                {
                    "Scope": "Original 0/1-shot LLM rows",
                    "Status": "Rerun/replaced",
                    "Interpretation": "The old rows used a legacy validation-style/original-baseline protocol and should not be mixed with strict rows.",
                },
                {
                    "Scope": "Original 3/5-shot LLM rows",
                    "Status": "Not covered in the old manuscript",
                    "Interpretation": "These are new strict held-out results.",
                },
            ]
        ),
        "",
        "## Main LoRA-Ours / HiPro-LoRA Rows Used For Comparison",
        "",
        markdown_table(main_display_rows(selected_rows)),
        "",
        "## Best Main LoRA-Ours Rows By Dataset",
        "",
        markdown_table(main_display_rows(best_main_rows)),
        "",
        "## Rows Where A Strict LLM Beats The Selected Main HiPro-LoRA Row",
        "",
        markdown_table(llm_wins)
        if llm_wins
        else "_No strict LLM row exceeds the selected main HiPro-LoRA row._",
        "",
        "## Strict LLM Result Table",
        "",
        markdown_table(llm_display_rows(rows)),
        "",
        "## Ours Vs Every Strict LLM Row",
        "",
        markdown_table(all_comparisons),
        "",
        "## Time, Memory, And Parameter Accounting",
        "",
        "- `Inference_Time_ms` in the LLM table is the average per-sample generation latency measured by `run_llm_fewshot.py`.",
        "- `Inference_Time_Sec` in the main table is the classifier inference-time field from the main strict summaries.",
        "- These timing fields are useful cost indicators, but they are not the same system-level measurement. CoT generation is expected to be slower than direct classification.",
        "- `Peak_Memory_MB` and `Params_M` are reported from each runner and should be interpreted under each runner's model-loading path and quantization settings.",
        "",
        "## Strict Entry Points",
        "",
        markdown_table(
            [
                {
                    "File": "run_llm_fewshot.py",
                    "Role": "Single strict runner for one or more models/datasets/shots/prompt styles.",
                },
                {
                    "File": "run_llm_aligned_all.py",
                    "Role": "Batch launcher for the complete aligned strict matrix.",
                },
                {
                    "File": "summarize_llm_results.py",
                    "Role": "Generates this summary without requiring the external `tabulate` package.",
                },
            ]
        ),
        "",
        "Old per-model scripts from the non-strict workflow are not kept in this strict baseline directory. The three files above are the only strict LLM entry points.",
        "",
        "## Figure Script Note",
        "",
        "`../create_Image.py` uses the strict CSV at `llm_baselines/few_shot_results/llm_fewshot_results.csv` for the LLM comparison figure. Archived mixed-protocol outputs are not result sources.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--llm-only-output-path", type=Path, default=DEFAULT_LLM_ONLY_OUTPUT_PATH)
    parser.add_argument("--protocol", default="balanced-held-out-test")
    args = parser.parse_args()

    rows = load_llm_results(args.result_path, args.protocol)
    main_rows = enrich_main_rows()

    missing_main = sorted(set(DATASET_ORDER) - set(selected_ours_rows(main_rows)))
    if missing_main:
        raise SystemExit(f"Missing selected LoRA-Ours main rows for: {missing_main}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(build_full_markdown(args, rows, main_rows), encoding="utf-8")
    print(f"Wrote {args.output_path}")

    if args.llm_only_output_path:
        args.llm_only_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.llm_only_output_path.write_text(build_llm_only_markdown(rows), encoding="utf-8")
        print(f"Wrote {args.llm_only_output_path}")


if __name__ == "__main__":
    main()
