import os, shutil, gc, torch, warnings, random, time, json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- Environment ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 1. Global Config (SST-5) ====================
MODEL_NAME = "roberta-base"
NUM_LABELS = 5
MAX_LENGTH = 128
DEFAULT_EPOCHS = 20 # 默认轮次
BATCH_SIZE = 64
RANDOM_SEEDS = [45, 123, 789, 2024, 1001]
PEFT_LR = 3e-4        

# Dataset Specific Configs
CONFIGS = {
    2300: {
        "train": {4: 1200, 3: 600, 2: 300, 1: 150, 0: 50}, 
        "eval_steps": 10, "loss_weight": 0.005, 
        "warmup_steps": 30, "lr_scale": 0.9, "grad_acc": 1,
        "epochs_override": 20, # N=2300 数据量够，20轮即可
        "smoothing": 0.1, "clamp_weights": True    
    },
    1150: {
        "train": {4: 600, 3: 300, 2: 150, 1: 80, 0: 20},
        "eval_steps": 10, "loss_weight": 0.0,    
        "warmup_steps": 20, "lr_scale": 1.0, "grad_acc": 1,              
        "epochs_override": 30, # [修改点] N=1150 单独改为 30 轮，防止欠拟合
        "smoothing": 0.05, "clamp_weights": True
    }
}
TAIL_CLASSES = [0, 1] 

# === [EXPERIMENT LIST: ONLY LoRA-Adv] ===
EXPERIMENTS = [
    # Reproduction: LoRA-Adv (From Paper)
    {
        "name": "LoRA-Adv", 
        "method": "LoRA-Adv",   
        "loss_type": "original", 
        "use_class_weight": True, 
        "peft_type": "lora",
        "adv_epsilon": 1e-3,     # 保持 1e-3
    }
]

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts", "sst5", "sst5_lora_adv_table2_strict")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MAIN_RESULTS_FILE = os.path.join(SCRIPT_DIR, "results", "sst5_lora_adv_table2_strict_results.csv")
IMG_DATA_DIR = os.path.join(ARTIFACT_DIR, "viz")
os.makedirs(IMG_DATA_DIR, exist_ok=True)
if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)

# ==================== Helper Functions ====================
def save_experiment_full_data(trainer, model, tokenizer, output_dir, file_prefix):
    with open(f"{output_dir}/{file_prefix}_history.json", "w") as f: 
        json.dump(trainer.state.log_history, f)
    
    dataloader = trainer.get_eval_dataloader()
    feats, labs, preds, logits_list, inputs_txt = [], [], [], [], []
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            
            if hasattr(model, "encoder") and hasattr(model.encoder, "forward"): 
                out_enc = model.encoder(inputs["input_ids"], inputs["attention_mask"])
                feat = out_enc.last_hidden_state[:, 0, :]
            elif hasattr(model, "model") and hasattr(model.model, "base_model"): 
                feat = model.model.base_model(inputs["input_ids"], inputs["attention_mask"]).last_hidden_state[:, 0, :]
            else: 
                feat = torch.zeros(inputs["input_ids"].size(0), 768)
            
            out = model(inputs["input_ids"], inputs["attention_mask"])
            logits = out["logits"] if isinstance(out, dict) else out.logits
            p = torch.argmax(logits, dim=-1)
            
            feats.append(feat.cpu().numpy())
            labs.append(inputs["labels"].cpu().numpy())
            preds.append(p.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            
            decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            inputs_txt.extend(decoded)
            
    np.savez(f"{output_dir}/{file_prefix}_viz.npz", 
             feats=np.vstack(feats), 
             labels=np.concatenate(labs), 
             preds=np.concatenate(preds), 
             logits=np.vstack(logits_list))
    
    df_cases = pd.DataFrame({"text": inputs_txt, "true_label": np.concatenate(labs), "pred_label": np.concatenate(preds)})
    df_cases.to_csv(f"{output_dir}/{file_prefix}_cases.csv", index=False)

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children(): 
        result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types) if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

# ==================== Model Class ====================
class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        target_modules = ["query", "key", "value"]
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=target_modules, 
            use_dora=False 
        )
        
        self.encoder = get_peft_model(AutoModel.from_pretrained(MODEL_NAME), peft_config)
        self.config = self.encoder.config
        self.config.num_labels = NUM_LABELS
        hs = self.encoder.config.hidden_size
        
        self.classifier_base = nn.Linear(hs, NUM_LABELS)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.encoder.base_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        else:
            hidden = self.encoder(input_ids, attention_mask).last_hidden_state
            
        cls_feat = hidden[:, 0, :]
        logits = self.classifier_base(cls_feat)
        return {"loss": None, "logits": logits}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def get_custom_dataset(df, config, seed):
    sampled = [df[df['label'] == l].sample(n=min(len(df[df['label'] == l]), c), random_state=seed) for l, c in config.items()]
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=-1)
    labels = eval_pred.label_ids
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    recalls = [report[str(i)]['recall'] for i in range(NUM_LABELS)]
    
    try: 
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except: 
        auc = 0.0
        
    metrics = {
        "macro_f1": f1_score(labels, preds, average="macro"), 
        "weighted_f1": f1_score(labels, preds, average="weighted"), 
        "accuracy": accuracy_score(labels, preds), 
        "balanced_acc": np.mean(recalls), 
        "g_mean": np.prod(recalls) ** (1/NUM_LABELS), 
        "auc": auc
    }
    for i in range(NUM_LABELS): 
        metrics[f"f1_class_{i}"] = report[str(i)]['f1-score']
    return metrics

def append_to_csv(filename, row_dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

# ==================== UnifiedTrainer ====================
class UnifiedTrainer(Trainer):
    def __init__(self, class_weights, smoothing=0.0, use_class_weight=True, adv_epsilon=None, method_name=None, **kwargs):
        super().__init__(**kwargs)
        self.use_class_weight = use_class_weight
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.label_smoothing = smoothing
        self.method_name = method_name
        self.adv_epsilon = adv_epsilon

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if "inputs_embeds" in inputs:
            outputs = model(inputs_embeds=inputs["inputs_embeds"], attention_mask=inputs["attention_mask"], labels=labels)
        else:
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        logits = outputs["logits"]
        
        weight_to_use = None
        if self.use_class_weight and self.class_weights is not None:
             if self.class_weights.device != logits.device: 
                 self.class_weights = self.class_weights.to(logits.device)
             weight_to_use = self.class_weights
        
        loss_fct = nn.CrossEntropyLoss(weight=weight_to_use, label_smoothing=self.label_smoothing)
        total_loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        return (total_loss, SequenceClassifierOutput(loss=total_loss, logits=logits)) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.method_name == "LoRA-Adv" and self.adv_epsilon is not None:
            # 1. Get Embeddings & Force Gradients
            if hasattr(model, "encoder") and hasattr(model.encoder, "base_model"):
                embed_layer = model.encoder.base_model.model.embeddings.word_embeddings
            elif hasattr(model, "model") and hasattr(model.model, "base_model"):
                embed_layer = model.model.base_model.embeddings.word_embeddings
            else:
                try: embed_layer = model.encoder.base_model.model.embeddings.word_embeddings
                except AttributeError: raise AttributeError("Could not find embedding layer.")

            input_ids = inputs["input_ids"]
            inputs_embeds = embed_layer(input_ids)
            
            # [CRITICAL FIX] Force gradients on the embeddings tensor
            inputs_embeds.requires_grad_(True) 
            inputs_embeds.retain_grad() 
            
            # 2. Clean Forward & Backward
            loss_clean = self.compute_loss(model, {"inputs_embeds": inputs_embeds, "attention_mask": inputs["attention_mask"], "labels": inputs["labels"]})
            if self.args.n_gpu > 1: loss_clean = loss_clean.mean()
            self.accelerator.backward(loss_clean, retain_graph=True) 
            
            # 3. Adversarial Perturbation
            grad = inputs_embeds.grad
            perturbation = self.adv_epsilon * grad.sign()
            adv_inputs_embeds = (inputs_embeds + perturbation).detach()
            
            # 4. Adversarial Forward & Backward
            loss_adv = self.compute_loss(model, {"inputs_embeds": adv_inputs_embeds, "attention_mask": inputs["attention_mask"], "labels": inputs["labels"]})
            if self.args.n_gpu > 1: loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            
            return (loss_clean + loss_adv).detach()
        else:
            loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1: loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach()

# ==================== Execution ====================
print(">>> Loading SST-5 Dataset...")
try: dataset_raw = load_dataset("SetFit/sst5")
except: dataset_raw = load_dataset("SetFit/sst5") 

full_df = pd.DataFrame(dataset_raw["train"]).dropna(subset=["text", "label"])
full_df["label"] = full_df["label"].astype(int)
val_source_df = pd.DataFrame(dataset_raw["validation"]).dropna(subset=["text", "label"])
val_source_df["label"] = val_source_df["label"].astype(int)
test_source_df = pd.DataFrame(dataset_raw["test"]).dropna(subset=["text", "label"])
test_source_df["label"] = test_source_df["label"].astype(int)
train_source_df = full_df.copy()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(ex): 
    return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)

print(f"\n{'='*80}\nSTRICT HELD-OUT EXPERIMENT START: SST-5 LoRA-Adv\n{'='*80}")

for N_SAMPLES in [2300, 1150]: 
    cfg = CONFIGS[N_SAMPLES]
    
    train_pool = train_source_df.copy()
    val_df = get_custom_dataset(val_source_df, {k: 80 for k in range(NUM_LABELS)}, 42)
    test_df = get_custom_dataset(test_source_df, {k: 80 for k in range(NUM_LABELS)}, 42)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])

    for exp in EXPERIMENTS:
        safe_name = exp['name'].replace('/', '_').replace('+', '_plus')
        
        for seed_idx, SEED in enumerate(RANDOM_SEEDS):
            print(f"\n[Run] N={N_SAMPLES} | {exp['name']} | Seed={SEED}")
            set_seed(SEED)
            
            train_df = get_custom_dataset(train_pool, cfg["train"], SEED)
            
            cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
            class_weights_np = cw
            if cfg['clamp_weights']: 
                class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
            
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
            
            model = UnifiedModel(exp).to(device)
            lr = PEFT_LR * cfg["lr_scale"]
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            output_dir_path = os.path.join(ARTIFACT_DIR, f"tmp_sst5_{N_SAMPLES}_{safe_name}_{SEED}")

            # 使用 Config 中定义的 epochs_override，如果没有则使用 DEFAULT_EPOCHS
            actual_epochs = cfg.get("epochs_override", DEFAULT_EPOCHS)

            trainer = UnifiedTrainer(
                class_weights=class_weights_np, 
                smoothing=cfg["smoothing"],
                use_class_weight=exp.get("use_class_weight", True),
                adv_epsilon=exp.get("adv_epsilon", None), 
                method_name=exp.get("method"),           
                model=model,
                args=TrainingArguments(
                    output_dir=output_dir_path, 
                    num_train_epochs=actual_epochs, # 动态设置 Epochs
                    per_device_train_batch_size=BATCH_SIZE, 
                    gradient_accumulation_steps=cfg["grad_acc"], 
                    learning_rate=lr, 
                    warmup_ratio=0.1, 
                    weight_decay=0.01, 
                    eval_strategy="steps", 
                    eval_steps=cfg["eval_steps"], 
                    save_steps=cfg["eval_steps"], 
                    save_total_limit=1, 
                    load_best_model_at_end=True, 
                    metric_for_best_model="macro_f1", 
                    fp16=True, 
                    report_to="none", 
                    logging_steps=5,
                    remove_unused_columns=False 
                ),
                train_dataset=train_ds, 
                eval_dataset=val_ds, 
                tokenizer=tokenizer, 
                data_collator=DataCollatorWithPadding(tokenizer), 
                compute_metrics=compute_metrics, 
                callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
            )
            
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            trainer.train()
            train_runtime = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            _ = trainer.evaluate()
            
            start_inf = time.time()
            pred_output = trainer.predict(test_ds)
            inf_time = time.time() - start_inf
            res = pred_output.metrics
            
            row = { 
                "Dataset": "SST-5", "N": N_SAMPLES, "Method": exp['name'], "Seed": SEED, 
                "Macro-F1": res['test_macro_f1'], "Weighted-F1": res['test_weighted_f1'], 
                "Accuracy": res['test_accuracy'], "Balanced_Acc": res['test_balanced_acc'], 
                "G-Mean": res['test_g_mean'], "AUC": res['test_auc'], 
                "Train_Time_Sec": train_runtime, "Inference_Time_Sec": inf_time, 
                "Peak_Memory_MB": peak_memory, "Params_M": num_params / 1e6 
            }
            for i in range(NUM_LABELS): 
                row[f"F1_Class_{i}"] = res[f"test_f1_class_{i}"]
            
            append_to_csv(MAIN_RESULTS_FILE, row)
            
            file_prefix = f"{safe_name}_N{N_SAMPLES}_seed{SEED}"
            trainer.eval_dataset = test_ds
            save_experiment_full_data(trainer, model, tokenizer, IMG_DATA_DIR, file_prefix)
            
            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()
            shutil.rmtree(output_dir_path, ignore_errors=True)

print(f"\n{'='*80}\nTRAINING DONE.\n{'='*80}")

# ==================== Final Auto-Summary Report ====================
def generate_final_summary(csv_path, tail_classes):
    if not os.path.exists(csv_path):
        print(f"!!! Error: Results file {csv_path} not found.")
        return

    print(f"\n{'='*80}\n>>> GENERATING FINAL SUMMARY REPORT...\n{'='*80}")
    try: df = pd.read_csv(csv_path)
    except: return

    if df.empty: return

    tail_cols = [f"F1_Class_{c}" for c in tail_classes]
    available_tail = [c for c in tail_cols if c in df.columns]
    if available_tail:
        df["Tail_F1"] = df[available_tail].mean(axis=1)
    
    target_metrics = [
        "Macro-F1", "Weighted-F1", "Accuracy", "Balanced_Acc", 
        "G-Mean", "AUC", "Tail_F1",
        "Train_Time_Sec", "Inference_Time_Sec", "Peak_Memory_MB"
    ]
    
    metrics = [m for m in target_metrics if m in df.columns]
    
    summary_rows = []
    grouped = df.groupby(["N", "Method"])

    for (n_val, method), group in grouped:
        row = {"N": n_val, "Method": method}
        best_idx = group["Macro-F1"].idxmax()
        row["Best (Seed/F1)"] = f"Seed {int(group.loc[best_idx, 'Seed'])}: {group.loc[best_idx, 'Macro-F1']:.4f}"

        for m in metrics:
            vals = group[m].dropna().tolist()
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                row[f"{m} (Mean±Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["N", "Method"])
    
    try: print("\n" + summary_df.to_markdown(index=False))
    except: print(summary_df)

    out_file = csv_path.replace(".csv", "_Summary.md")
    with open(out_file, "w") as f:
        f.write(f"# Final Experiment Summary: LoRA-Adv Reproduction\n")
        f.write(f"Generated Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        try: f.write(summary_df.to_markdown(index=False))
        except: f.write(str(summary_df))
    print(f"\n>>> Full Summary Saved to: {out_file}")

if __name__ == "__main__":
    generate_final_summary(MAIN_RESULTS_FILE, TAIL_CLASSES)
import shutil
import os

def zip_working_directory():
    # 源目录
    source_dir = '/kaggle/working/'
    # 输出文件名（注意：不需要加 .zip 后缀，shutil 会自动添加）
    output_filename = '/kaggle/working/working_backup'
    
    print(f"正在打包 {source_dir} ...")
    
    try:
        # 创建压缩包
        # format='zip': 压缩格式
        # root_dir: 要压缩的根目录
        shutil.make_archive(output_filename, 'zip', source_dir)
        print(f"✅ 打包成功！")
        print(f"文件位置: {output_filename}.zip")
        print(f"文件大小: {os.path.getsize(output_filename + '.zip') / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ 打包失败: {e}")

# 执行打包
zip_working_directory()
