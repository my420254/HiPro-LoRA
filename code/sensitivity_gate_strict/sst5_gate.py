import os, shutil, gc, torch, warnings, random, time, json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback,
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
EPOCHS = 20
BATCH_SIZE = 64  # <--- [已修复] 补上了 BATCH_SIZE
RANDOM_SEEDS = [45, 123, 789, 2024, 1001]
FULL_LR = 2e-5        
PEFT_LR = 3e-4        
# 0              1            2               3               4
# very neg      neg          neutral         pos           very pos
# Dataset Specific Configs
CONFIGS = {
    2300: {
        "train": {4: 1200, 3: 600, 2: 300, 1: 150, 0: 50}, 
        "eval_steps": 10, "memory_size": 200, "temperature": 0.3, "loss_weight": 0.005, 
        "warmup_steps": 30, "tail_weight": 1.0, "lr_scale": 0.9, "grad_acc": 1,
        "fusion_init": -1.8, "smoothing": 0.1, "clamp_weights": True    
    },
    1150: {
        "train": {4: 600, 3: 300, 2: 150, 1: 80, 0: 20},
        "eval_steps": 10, "memory_size": 200, "temperature": 0.15, "loss_weight": 0.0,   
        "warmup_steps": 20, "tail_weight": 1.0, "lr_scale": 1.0, "grad_acc": 1,              
        "fusion_init": -2.0, "smoothing": 0.05, "clamp_weights": True
    }
}
TAIL_CLASSES = [0, 1] 

EXPERIMENTS = [
    {"name": "LoRA-Ours", "method": "peft", "loss_type": "original", "use_class_weight": True, "peft_type": "lora", "hsp": True, "memory_bank": True},
]

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # HiPro-LoRA root
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
GATE_LOG_DIR = os.path.join(SCRIPT_DIR, "gate_logs", "sst5")
TABLE2_REF_FILE = os.path.join(PROJECT_ROOT, "table2_strict", "results", "sst5_main_table2_strict_results.csv")
TARGET_N = int(os.environ.get("HIPRO_GATE_N", "1150"))
TARGET_METHOD = os.environ.get("HIPRO_GATE_METHOD", "LoRA-Ours")
TARGET_SEEDS = [int(os.environ["HIPRO_GATE_SEED"])] if os.environ.get("HIPRO_GATE_SEED") else RANDOM_SEEDS
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GATE_LOG_DIR, exist_ok=True)
MAIN_RESULTS_FILE = os.path.join(RESULTS_DIR, f"sst5_gate_N{TARGET_N}_seed{TARGET_SEEDS[0] if len(TARGET_SEEDS) == 1 else 'all'}.csv")
if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)

# ==================== Helper & Classes ====================
def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children(): result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types) if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

class MemoryBank(nn.Module):
    def __init__(self, feature_dim=128, num_classes=5, memory_size=600, temperature=0.3, tail_classes=[0, 1], tail_weight=1.3, warmup_steps=10, min_samples=5):
        super().__init__(); self.feature_dim = feature_dim; self.num_classes = num_classes; self.temperature = temperature
        self.tail_classes = tail_classes; self.tail_weight = tail_weight; self.warmup_steps = warmup_steps; self.min_samples = min_samples; self.current_step = 0
        capacity = memory_size // num_classes
        for c in range(num_classes): self.register_buffer(f'memory_bank_{c}', torch.randn(capacity, feature_dim))
        self.register_buffer('bank_ptrs', torch.zeros(num_classes, dtype=torch.long)); self.register_buffer('bank_sizes', torch.zeros(num_classes, dtype=torch.long))
    def get_memory_bank(self, class_id): return getattr(self, f'memory_bank_{class_id}')
    def set_memory_bank(self, class_id, data, start_idx, end_idx): getattr(self, f'memory_bank_{class_id}')[start_idx:end_idx] = data
    @torch.no_grad()
    def update_memory_bank(self, features, labels):
        if self.current_step < self.warmup_steps: return
        features = F.normalize(features.detach().clone(), dim=1); labels = labels.detach().clone()
        for c in range(self.num_classes):
            mask = (labels == c);
            if not mask.any(): continue
            feats_c = features[mask].clone(); n = feats_c.size(0); bank = self.get_memory_bank(c); ptr = self.bank_ptrs[c].item(); cap = bank.size(0)
            if ptr + n <= cap: self.set_memory_bank(c, feats_c, ptr, ptr + n); self.bank_ptrs[c] = (ptr + n) % cap
            else:
                rem = cap - ptr
                if n <= rem:
                    self.set_memory_bank(c, feats_c, ptr, ptr + n)
                else:
                    self.set_memory_bank(c, feats_c[:rem], ptr, cap)
                    overflow = n - rem
                    take = min(overflow, cap)
                    if take > 0:
                        self.set_memory_bank(c, feats_c[rem:rem+take], 0, take)
                self.bank_ptrs[c] = min((ptr + n) % cap, cap - 1) if cap > 0 else 0
            self.bank_sizes[c] = min(self.bank_sizes[c] + n, cap)
    def forward(self, features, labels):
        self.current_step += 1;
        if self.current_step <= self.warmup_steps: return torch.tensor(0.0, device=features.device, requires_grad=True)
        features_norm = F.normalize(features, dim=1); total_loss = 0.0; valid = 0
        for i in range(features.size(0)):
            feat = features_norm[i]; label = labels[i].item(); pos = self.get_memory_bank(label)[:self.bank_sizes[label]].detach().clone()
            if pos.size(0) < self.min_samples: continue
            negs = [self.get_memory_bank(c)[:self.bank_sizes[c]].detach().clone() for c in range(self.num_classes) if c != label and self.bank_sizes[c] >= self.min_samples]
            if not negs: continue
            neg_feats = torch.cat(negs, dim=0); logits = torch.cat([torch.matmul(feat.unsqueeze(0), pos.t()) / self.temperature, torch.matmul(feat.unsqueeze(0), neg_feats.t()) / self.temperature], dim=1)
            total_loss += (self.tail_weight if label in self.tail_classes else 1.0) * F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=features.device)); valid += 1
        return total_loss / valid if valid > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)

class HierarchicalSmartPooling(nn.Module):
    def __init__(self, hs, dr=0.1):
        super().__init__(); self.attn = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Softmax(dim=1)); self.fusion = nn.Sequential(nn.Linear(hs*3, hs*2), nn.LayerNorm(hs*2), nn.GELU(), nn.Dropout(dr), nn.Linear(hs*2, hs))
    def forward(self, x, m):
        w = self.attn(x).masked_fill(m.unsqueeze(-1)==0, -1e9); w = F.softmax(w, dim=1)
        return self.fusion(torch.cat([torch.sum(x*w, 1), torch.sum(x*m.unsqueeze(-1), 1)/m.sum(1, keepdim=True).clamp(min=1e-9), x.masked_fill(m.unsqueeze(-1)==0, -1e9).max(1)[0]], dim=1))

class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.is_peft = (cfg["method"] == "peft")
        if not self.is_peft: self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS); self.config = self.model.config
        else:
            peft_type = cfg.get("peft_type", "lora"); target_modules = ["query", "key", "value"]
            use_dora = True if peft_type == "dora" else False
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules, use_dora=use_dora)
            self.encoder = get_peft_model(AutoModel.from_pretrained(MODEL_NAME), peft_config); self.config = self.encoder.config; self.config.num_labels = NUM_LABELS; hs = self.encoder.config.hidden_size
            self.classifier_base = nn.Linear(hs, NUM_LABELS)
            if cfg["hsp"]: self.hsp_module = HierarchicalSmartPooling(hs); self.classifier_hsp = nn.Linear(hs, NUM_LABELS); nn.init.constant_(self.classifier_hsp.weight, 0); nn.init.constant_(self.classifier_hsp.bias, 0); self.fusion_weight = nn.Parameter(torch.tensor([cfg.get("fusion_init", 0.1)]))
            else: self.hsp_module = None
            if cfg["memory_bank"]: self.projector = nn.Sequential(nn.Linear(hs, hs), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hs, 128))
            else: self.projector = None
    def forward(self, input_ids, attention_mask, labels=None):
        if not self.is_peft: return {"loss": None, "logits": self.model(input_ids, attention_mask, labels=labels).logits, "proj_features": None}
        hidden = self.encoder(input_ids, attention_mask).last_hidden_state; cls_feat = hidden[:, 0, :]; logits = self.classifier_base(cls_feat)
        if self.hsp_module: logits = logits + torch.sigmoid(self.fusion_weight) * self.classifier_hsp(self.hsp_module(hidden, attention_mask))
        return {"loss": None, "logits": logits, "proj_features": self.projector(cls_feat) if self.projector else None}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_custom_dataset(df, config, seed):
    sampled = [df[df['label'] == l].sample(n=min(len(df[df['label'] == l]), c), random_state=seed) for l, c in config.items()]
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

def compute_metrics(eval_pred):
    logits = eval_pred.predictions; preds = np.argmax(logits, axis=-1); labels = eval_pred.label_ids
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    recalls = [report[str(i)]['recall'] for i in range(NUM_LABELS)]
    try: probs = F.softmax(torch.tensor(logits), dim=-1).numpy(); auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except: auc = 0.0
    metrics = {"macro_f1": f1_score(labels, preds, average="macro"), "weighted_f1": f1_score(labels, preds, average="weighted"), "accuracy": accuracy_score(labels, preds), "balanced_acc": np.mean(recalls), "g_mean": np.prod(recalls) ** (1/NUM_LABELS), "auc": auc}
    for i in range(NUM_LABELS): metrics[f"f1_class_{i}"] = report[str(i)]['f1-score']
    return metrics

def append_to_csv(filename, row_dict):
    df = pd.DataFrame([row_dict]); df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

def get_gate_values(model):
    raw = getattr(model, "fusion_weight", None)
    if raw is None:
        return None, None
    raw_value = float(raw.detach().float().cpu().view(-1)[0].item())
    return raw_value, float(torch.sigmoid(torch.tensor(raw_value)).item())

def append_gate_snapshot(filename, model, dataset_name, n_samples, method, seed, phase, state=None, metrics=None):
    gate_raw, gate_sigmoid = get_gate_values(model)
    row = {
        "Dataset": dataset_name,
        "N": n_samples,
        "Method": method,
        "Seed": seed,
        "phase": phase,
        "step": int(state.global_step) if state is not None else None,
        "epoch": float(state.epoch) if state is not None and state.epoch is not None else None,
        "gate_raw": gate_raw,
        "gate_sigmoid": gate_sigmoid,
    }
    append_to_csv(filename, row)

class GateTraceCallback(TrainerCallback):
    def __init__(self, dataset_name, n_samples, method, seed, filename):
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.filename = filename

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        append_gate_snapshot(self.filename, model, self.dataset_name, self.n_samples, self.method, self.seed, "train_begin", state)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        append_gate_snapshot(self.filename, model, self.dataset_name, self.n_samples, self.method, self.seed, "eval", state, metrics)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        append_gate_snapshot(self.filename, model, self.dataset_name, self.n_samples, self.method, self.seed, "train_end", state)

def table2_reference(n_samples, method, seed):
    if not os.path.exists(TABLE2_REF_FILE):
        return {}
    df = pd.read_csv(TABLE2_REF_FILE)
    m = df[(df["N"].astype(int) == int(n_samples)) & (df["Method"].astype(str) == str(method)) & (df["Seed"].astype(int) == int(seed))]
    if m.empty:
        return {}
    ref = m.iloc[0].to_dict()
    keep = ["Macro-F1", "Weighted-F1", "Accuracy", "Balanced_Acc", "G-Mean", "AUC"]
    return {f"Table2Ref_{k}": ref[k] for k in keep if k in ref}

class UnifiedTrainer(Trainer):
    def __init__(self, loss_type, class_weights, cls_num_list, memory_loss, loss_weight, is_peft, smoothing, use_class_weight=True, **kwargs):
        super().__init__(**kwargs); self.loss_type = loss_type; self.use_class_weight = use_class_weight; self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.cls_num_list = cls_num_list; self.memory_loss_module = memory_loss; self.aux_loss_weight = loss_weight; self.is_peft = is_peft; self.label_smoothing = smoothing; self.current_epoch = 0
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels"); outputs = model(inputs["input_ids"], inputs["attention_mask"], labels); logits = outputs["logits"]
        weight_to_use = None
        if self.use_class_weight and self.class_weights is not None:
             if self.class_weights.device != logits.device: self.class_weights = self.class_weights.to(logits.device)
             weight_to_use = self.class_weights
        loss_fct = nn.CrossEntropyLoss(weight=weight_to_use, label_smoothing=self.label_smoothing); total_loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        if self.is_peft and self.memory_loss_module is not None and outputs.get("proj_features") is not None:
            proj_features = outputs["proj_features"]; loss_mb = self.memory_loss_module(proj_features, labels); total_loss += self.aux_loss_weight * loss_mb
            with torch.no_grad(): self.memory_loss_module.update_memory_bank(proj_features, labels)
        return (total_loss, SequenceClassifierOutput(loss=total_loss, logits=logits)) if return_outputs else total_loss
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm]); decay_parameters = [name for name in decay_parameters if "bias" not in name]; optimizer_grouped_parameters = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad: continue
                if "fusion_weight" in n: optimizer_grouped_parameters.append({"params": [p], "weight_decay": 0.0, "lr": self.args.learning_rate * 5})
                else: optimizer_grouped_parameters.append({"params": [p], "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0, "lr": self.args.learning_rate})
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args); self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    def on_epoch_begin(self, args, state, control, **kwargs): self.current_epoch = state.epoch

# ==================== 4. 数据 & 实验 A ====================
print(">>> Loading SST-5 Dataset...")
try: dataset_raw = load_dataset("SetFit/sst5")
except: dataset_raw = load_dataset("SetFit/sst5") 
full_df = pd.DataFrame(dataset_raw["train"]).dropna(subset=["text", "label"])
full_df["label"] = full_df["label"].astype(int)
def to_standard_df(split_name):
    df = pd.DataFrame(dataset_raw[split_name])
    if "content" in df.columns:
        df = df.rename(columns={"content": "text"})
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]

train_source_df = full_df.copy()
val_source_df = to_standard_df("validation")
test_source_df = to_standard_df("test")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(ex): return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)
if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)

print(f"\n{'='*80}\nPART A: TABLE2-ALIGNED GATE TRACE (held-out test)\n{'='*80}")
print(f">>> Gate run: dataset=SST-5 | N={TARGET_N} | method={TARGET_METHOD} | seeds={TARGET_SEEDS}")
if TARGET_N not in CONFIGS:
    raise ValueError(f"Unsupported TARGET_N={TARGET_N}; available={sorted(CONFIGS)}")
for N_SAMPLES in [TARGET_N]: 
    cfg = CONFIGS[N_SAMPLES]
    train_pool = train_source_df.copy()
    val_df = get_custom_dataset(val_source_df, {k: 80 for k in range(NUM_LABELS)}, 42)
    test_df = get_custom_dataset(test_source_df, {k: 80 for k in range(NUM_LABELS)}, 42)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])

    for exp in EXPERIMENTS:
        if exp["name"] != TARGET_METHOD:
            continue
        safe_name = exp['name'].replace('/', '_').replace('+', '_plus')
        for seed_idx, SEED in enumerate(TARGET_SEEDS):
            print(f"\n[Part A] N={N_SAMPLES} | {exp['name']} | Seed={SEED}")
            set_seed(SEED)
            train_df = get_custom_dataset(train_pool, cfg["train"], SEED)
            cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
            class_weights_np = cw
            if cfg['clamp_weights']: class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
            cls_num_list = [len(train_df[train_df['label'] == i]) for i in range(NUM_LABELS)]
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
            
            current_cfg = exp.copy(); current_cfg["fusion_init"] = cfg["fusion_init"]
            model = UnifiedModel(current_cfg).to(device)
            lr = FULL_LR if exp["method"] == "full_ft" else PEFT_LR * cfg["lr_scale"]
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            output_dir_path = f"./tmp_sst5_gate_{N_SAMPLES}_{safe_name}_{SEED}"
            gate_log_file = os.path.join(GATE_LOG_DIR, f"sst5_N{N_SAMPLES}_{safe_name}_seed{SEED}_gate.csv")
            if os.path.exists(gate_log_file):
                os.remove(gate_log_file)

            trainer = UnifiedTrainer(
                loss_type=exp["loss_type"],
                class_weights=class_weights_np, 
                cls_num_list=cls_num_list,
                memory_loss=MemoryBank(128, NUM_LABELS, cfg["memory_size"], cfg["temperature"], TAIL_CLASSES, cfg["tail_weight"], cfg["warmup_steps"], 5).to(device) if exp["memory_bank"] else None,
                loss_weight=cfg["loss_weight"], 
                is_peft=(exp["method"] == "peft"), 
                model=model,
                use_class_weight=exp.get("use_class_weight", True),
                args=TrainingArguments(output_dir=output_dir_path, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=cfg["grad_acc"], learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01, eval_strategy="steps", eval_steps=cfg["eval_steps"], save_steps=cfg["eval_steps"], save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1", fp16=True, report_to="none", logging_steps=5),
                train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=8), GateTraceCallback("SST-5", N_SAMPLES, exp['name'], SEED, gate_log_file)], 
                smoothing=cfg["smoothing"]
            )
            
            torch.cuda.reset_peak_memory_stats(); start_time = time.time(); trainer.train()
            train_runtime = time.time() - start_time; peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024; _ = trainer.evaluate()
            start_inf = time.time(); pred_output = trainer.predict(test_ds); inf_time = time.time() - start_inf
            res = pred_output.metrics
            append_gate_snapshot(gate_log_file, model, "SST-5", N_SAMPLES, exp['name'], SEED, "best_model_after_predict", trainer.state, res)
            gate_raw_final, gate_sigmoid_final = get_gate_values(model)
            
            row = { "Dataset": "SST-5", "N": N_SAMPLES, "Method": exp['name'], "Seed": SEED, "Macro-F1": res['test_macro_f1'], "Weighted-F1": res['test_weighted_f1'], "Accuracy": res['test_accuracy'], "Balanced_Acc": res['test_balanced_acc'], "G-Mean": res['test_g_mean'], "AUC": res['test_auc'], "Train_Time_Sec": train_runtime, "Inference_Time_Sec": inf_time, "Peak_Memory_MB": peak_memory, "Params_M": num_params / 1e6 }
            for i in range(NUM_LABELS): row[f"F1_Class_{i}"] = res[f"test_f1_class_{i}"]
            row.update({"Run_Macro-F1": row["Macro-F1"], "Gate_Raw_Final": gate_raw_final, "Gate_Sigmoid_Final": gate_sigmoid_final, "Gate_Log_File": gate_log_file})
            row.update(table2_reference(N_SAMPLES, exp['name'], SEED))
            if "Table2Ref_Macro-F1" in row:
                row["Macro-F1_Diff_vs_Table2Ref"] = row["Run_Macro-F1"] - row["Table2Ref_Macro-F1"]
            append_to_csv(MAIN_RESULTS_FILE, row)
            del model, trainer; torch.cuda.empty_cache(); gc.collect(); shutil.rmtree(output_dir_path, ignore_errors=True)


print("\nStrict held-out script stops after the main experiments.")
