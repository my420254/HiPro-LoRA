import os, shutil, gc, torch, warnings, random, time, json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback,
    AutoModelForSequenceClassification, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- 环境变量 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 1. Global Config (SMP2020) ====================
MODEL_NAME = "hfl/chinese-macbert-base"
NUM_LABELS = 6
MAX_LENGTH = 128
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_SEEDS = [45, 123, 789, 1001, 2024]
FULL_LR = 2e-5
PEFT_LR = 3e-4

CONFIGS = {
    1000: {
        "train": {3: 450, 2: 250, 0: 120, 4: 100, 1: 60, 5: 20},
        "eval_steps": 10, "memory_size": 1200, "temperature": 0.2, "loss_weight": 0.2,
        "warmup_steps": 5, "tail_weight": 3.5, "lr_scale": 1.0, "grad_acc": 2,
        "fusion_init": -2.0, "smoothing": 0.05, "clamp_weights": True
    },
    2000: {
        "train": {3: 1000, 2: 500, 0: 200, 4: 150, 1: 100, 5: 50},
        "eval_steps": 10, "memory_size": 1200, "temperature": 0.15, "loss_weight": 0.2,
        "warmup_steps": 30, "tail_weight": 4.0, "lr_scale": 1.0, "grad_acc": 1,
        "fusion_init": 0.0, "smoothing": 0.1, "clamp_weights": True
    }
}
TAIL_CLASSES = [1, 5]

# ✅ 只跑 LoRA-Ours
EXPERIMENTS = [{'name': 'LoRA-Ours', 'method': 'peft', 'loss_type': 'original', 'use_class_weight': True, 'peft_type': 'lora', 'hsp': True, 'memory_bank': True}]

# ✅ 敏感性：MemSize + TailWeight
SENS_MEM_SIZES = [200, 600, 1200, 2000]
SENS_TAIL_WEIGHTS = [2.0, 3.0, 3.5, 4.0]
SENS_TEMPS = [0.05, 0.1, 0.2, 0.4]
SENS_LOSS_WEIGHTS = [0.05, 0.1, 0.2, 0.3]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR is now the sensitivity_gate_strict directory
BASE_DIR = SCRIPT_DIR
MAIN_RESULTS_FILE = os.path.join(SCRIPT_DIR, "results", "smp2020_sensitivity_main.csv")
SENSITIVITY_FILE = os.path.join(SCRIPT_DIR, "results", "smp2020_sensitivity.csv")
GATE_LOG_DIR      = os.path.join(SCRIPT_DIR, "gate_logs", "smp2020_sensitivity")          # ✅ 门控过程日志目录
IMG_DATA_DIR = os.path.join(SCRIPT_DIR, "viz", "smp2020_sensitivity")
os.makedirs(IMG_DATA_DIR, exist_ok=True)
os.makedirs(GATE_LOG_DIR, exist_ok=True)

# ==================== 2. Gate Monitor Callback ====================
class GateMonitorCallback(TrainerCallback):
    """每次 evaluate 时记录一次 σ(fusion_weight) 的当前值"""
    def __init__(self, model, log_path):
        self.model    = model
        self.log_path = log_path
        self.records  = []

    def on_evaluate(self, args, state, control, **kwargs):
        gate_val = torch.sigmoid(self.model.fusion_weight).item()
        self.records.append({
            "step":  state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "gate_sigmoid": gate_val
        })

    def on_train_end(self, args, state, control, **kwargs):
        pd.DataFrame(self.records).to_csv(self.log_path, index=False)
        print(f"  [GateMonitor] Saved {len(self.records)} checkpoints → {self.log_path}")

# ==================== 3. Helper Classes ====================
def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types)
                   if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

class MemoryBank(nn.Module):
    def __init__(self, feature_dim=128, num_classes=6, memory_size=600, temperature=0.3,
                 tail_classes=[1, 5], tail_weight=1.3, warmup_steps=10, min_samples=5):
        super().__init__()
        self.feature_dim = feature_dim; self.num_classes = num_classes
        self.temperature = temperature; self.tail_classes = tail_classes
        self.tail_weight = tail_weight; self.warmup_steps = warmup_steps
        self.min_samples = min_samples; self.current_step = 0
        capacity = memory_size // num_classes
        for c in range(num_classes):
            self.register_buffer(f'memory_bank_{c}', torch.randn(capacity, feature_dim))
        self.register_buffer('bank_ptrs',  torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('bank_sizes', torch.zeros(num_classes, dtype=torch.long))

    def get_memory_bank(self, class_id): return getattr(self, f'memory_bank_{class_id}')
    def set_memory_bank(self, class_id, data, s, e): getattr(self, f'memory_bank_{class_id}')[s:e] = data

    @torch.no_grad()
    def update_memory_bank(self, features, labels):
        if self.current_step < self.warmup_steps: return
        features = F.normalize(features.detach().clone(), dim=1)
        labels   = labels.detach().clone()
        for c in range(self.num_classes):
            mask = (labels == c)
            if not mask.any(): continue
            feats_c = features[mask].clone(); n = feats_c.size(0)
            bank = self.get_memory_bank(c); ptr = self.bank_ptrs[c].item(); cap = bank.size(0)
            if ptr + n <= cap:
                self.set_memory_bank(c, feats_c, ptr, ptr + n)
                self.bank_ptrs[c] = (ptr + n) % cap
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
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        features_norm = F.normalize(features, dim=1)
        total_loss = 0.0; valid = 0
        for i in range(features.size(0)):
            feat  = features_norm[i]; label = labels[i].item()
            pos   = self.get_memory_bank(label)[:self.bank_sizes[label]].detach().clone()
            if pos.size(0) < self.min_samples: continue
            negs  = [self.get_memory_bank(c)[:self.bank_sizes[c]].detach().clone()
                     for c in range(self.num_classes)
                     if c != label and self.bank_sizes[c] >= self.min_samples]
            if not negs: continue
            neg_feats = torch.cat(negs, dim=0)
            logits_mb = torch.cat([
                torch.matmul(feat.unsqueeze(0), pos.t())       / self.temperature,
                torch.matmul(feat.unsqueeze(0), neg_feats.t()) / self.temperature
            ], dim=1)
            total_loss += (self.tail_weight if label in self.tail_classes else 1.0) * \
                          F.cross_entropy(logits_mb, torch.zeros(1, dtype=torch.long, device=features.device))
            valid += 1
        return total_loss / valid if valid > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)

class HierarchicalSmartPooling(nn.Module):
    def __init__(self, hs, dr=0.1):
        super().__init__()
        self.attn   = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Softmax(dim=1))
        self.fusion = nn.Sequential(nn.Linear(hs*3, hs*2), nn.LayerNorm(hs*2),
                                    nn.GELU(), nn.Dropout(dr), nn.Linear(hs*2, hs))
    def forward(self, x, m):
        w = self.attn(x).masked_fill(m.unsqueeze(-1) == 0, -1e9)
        w = F.softmax(w, dim=1)
        return self.fusion(torch.cat([
            torch.sum(x * w, 1),
            torch.sum(x * m.unsqueeze(-1), 1) / m.sum(1, keepdim=True).clamp(min=1e-9),
            x.masked_fill(m.unsqueeze(-1) == 0, -1e9).max(1)[0]
        ], dim=1))

class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg; self.is_peft = (cfg["method"] == "peft")
        if not self.is_peft:
            self.model  = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
            self.config = self.model.config
        else:
            use_dora    = cfg.get("peft_type", "lora") == "dora"
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["query", "key", "value"],
                                     use_dora=use_dora)
            self.encoder = get_peft_model(AutoModel.from_pretrained(MODEL_NAME), peft_config)
            self.config  = self.encoder.config; self.config.num_labels = NUM_LABELS
            hs = self.encoder.config.hidden_size
            self.classifier_base = nn.Linear(hs, NUM_LABELS)
            if cfg["hsp"]:
                self.hsp_module      = HierarchicalSmartPooling(hs)
                self.classifier_hsp  = nn.Linear(hs, NUM_LABELS)
                nn.init.constant_(self.classifier_hsp.weight, 0)
                nn.init.constant_(self.classifier_hsp.bias,   0)
                self.fusion_weight   = nn.Parameter(torch.tensor([cfg.get("fusion_init", 0.1)]))
            else:
                self.hsp_module = None
            self.projector = nn.Sequential(nn.Linear(hs, hs), nn.ReLU(),
                                           nn.Dropout(0.1), nn.Linear(hs, 128)) if cfg["memory_bank"] else None

    def forward(self, input_ids, attention_mask, labels=None):
        if not self.is_peft:
            return {"loss": None,
                    "logits": self.model(input_ids, attention_mask, labels=labels).logits,
                    "proj_features": None}
        hidden   = self.encoder(input_ids, attention_mask).last_hidden_state
        cls_feat = hidden[:, 0, :]
        logits   = self.classifier_base(cls_feat)
        if self.hsp_module:
            logits = logits + torch.sigmoid(self.fusion_weight) * \
                     self.classifier_hsp(self.hsp_module(hidden, attention_mask))
        return {"loss": None, "logits": logits,
                "proj_features": self.projector(cls_feat) if self.projector else None}

class UnifiedTrainer(Trainer):
    def __init__(self, loss_type, class_weights, cls_num_list, memory_loss,
                 loss_weight, is_peft, smoothing, use_class_weight=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_type          = loss_type
        self.use_class_weight   = use_class_weight
        self.class_weights      = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.cls_num_list       = cls_num_list
        self.memory_loss_module = memory_loss
        self.aux_loss_weight    = loss_weight
        self.is_peft            = is_peft
        self.label_smoothing    = smoothing
        self.current_epoch      = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        # ✅ 显式传参，避免多余字段进入 forward
        outputs = model(inputs["input_ids"], inputs["attention_mask"], labels)
        logits  = outputs["logits"]
        weight_to_use = None
        if self.use_class_weight and self.class_weights is not None:
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            weight_to_use = self.class_weights
        loss_fct    = nn.CrossEntropyLoss(weight=weight_to_use, label_smoothing=self.label_smoothing)
        total_loss  = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        if self.is_peft and self.memory_loss_module is not None and outputs.get("proj_features") is not None:
            proj_features = outputs["proj_features"]
            loss_mb       = self.memory_loss_module(proj_features, labels)
            total_loss   += self.aux_loss_weight * loss_mb
            with torch.no_grad():
                self.memory_loss_module.update_memory_bank(proj_features, labels)
        return (total_loss, SequenceClassifierOutput(loss=total_loss, logits=logits)) if return_outputs else total_loss

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [n for n in decay_parameters if "bias" not in n]
            groups = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad: continue
                if "fusion_weight" in n:
                    groups.append({"params": [p], "weight_decay": 0.0, "lr": self.args.learning_rate * 5})
                else:
                    groups.append({"params": [p],
                                   "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0,
                                   "lr": self.args.learning_rate})
            opt_cls, opt_kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer  = opt_cls(groups, **opt_kw)
        return self.optimizer

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch

# ==================== 4. Utility Functions ====================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_custom_dataset(df, config, seed):
    sampled = [df[df['label'] == l].sample(n=min(len(df[df['label'] == l]), c), random_state=seed)
               for l, c in config.items()]
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

def compute_metrics(eval_pred):
    logits = eval_pred.predictions; preds = np.argmax(logits, axis=-1); labels = eval_pred.label_ids
    report  = classification_report(labels, preds, output_dict=True, zero_division=0)
    recalls = [report[str(i)]['recall'] for i in range(NUM_LABELS)]
    try:
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        auc   = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except: auc = 0.0
    metrics = {
        "macro_f1":    f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy":    accuracy_score(labels, preds),
        "balanced_acc": np.mean(recalls),
        "g_mean":      np.prod(recalls) ** (1 / NUM_LABELS),
        "auc":         auc
    }
    for i in range(NUM_LABELS): metrics[f"f1_class_{i}"] = report[str(i)]['f1-score']
    return metrics

def append_to_csv(filename, row_dict):
    pd.DataFrame([row_dict]).to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

def make_training_args(output_dir, cfg, lr):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=True,
        report_to="none",
        logging_steps=5
    )

# ==================== 5. Data Loading ====================
print(">>> Loading SMP2020 Dataset...")
dataset_raw = load_dataset("Um1neko/smp2020")
full_df = pd.DataFrame(dataset_raw["train"])
if "content" in full_df.columns: full_df = full_df.rename(columns={"content": "text"})
full_df = full_df.dropna(subset=["text", "label"])
full_df["label"] = full_df["label"].astype(int)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(ex): return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)
if os.path.exists(SENSITIVITY_FILE):  os.remove(SENSITIVITY_FILE)

# ==================== PART A: Main Experiment ====================
print(f"\n{'='*80}\nPART A: MAIN EXPERIMENT (LoRA-Ours Only)\n{'='*80}")
for N_SAMPLES in [2000]:
    cfg = CONFIGS[N_SAMPLES]
    train_pool, val_pool = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
    val_df = get_custom_dataset(val_pool, {k: 50 for k in range(NUM_LABELS)}, 42)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(
        ["input_ids", "attention_mask", "label"]
    )

    for exp in EXPERIMENTS:
        safe_name = exp['name'].replace('/', '_').replace('+', '_plus')
        for SEED in RANDOM_SEEDS:
            print(f"\n[Part A] N={N_SAMPLES} | {exp['name']} | Seed={SEED}")
            set_seed(SEED)
            train_df = get_custom_dataset(train_pool, cfg["train"], SEED)
            cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
            class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy() \
                if cfg['clamp_weights'] else cw
            cls_num_list = [len(train_df[train_df['label'] == i]) for i in range(NUM_LABELS)]
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(
                ["input_ids", "attention_mask", "label"]
            )
            current_cfg = exp.copy(); current_cfg["fusion_init"] = cfg["fusion_init"]
            model = UnifiedModel(current_cfg).to(device)
            lr    = FULL_LR if exp["method"] == "full_ft" else PEFT_LR * cfg["lr_scale"]
            num_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            output_dir   = f"./tmp_smp_{N_SAMPLES}_{safe_name}_{SEED}"

            # ✅ 门控过程日志路径
            gate_log_path = f"{GATE_LOG_DIR}/gate_N{N_SAMPLES}_{safe_name}_seed{SEED}.csv"
            gate_cb = GateMonitorCallback(model, gate_log_path)

            trainer = UnifiedTrainer(
                loss_type=exp["loss_type"], class_weights=class_weights_np,
                cls_num_list=cls_num_list,
                memory_loss=MemoryBank(128, NUM_LABELS, cfg["memory_size"], cfg["temperature"],
                                       TAIL_CLASSES, cfg["tail_weight"], cfg["warmup_steps"], 5).to(device),
                loss_weight=cfg["loss_weight"], is_peft=(exp["method"] == "peft"),
                smoothing=cfg["smoothing"], use_class_weight=exp.get("use_class_weight", True),
                model=model, train_dataset=train_ds, eval_dataset=val_ds,
                tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                args=make_training_args(output_dir, cfg, lr),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=8), gate_cb]  # ✅ 加入门控监控
            )

            torch.cuda.reset_peak_memory_stats()
            start_time = time.time(); trainer.train()
            train_runtime = time.time() - start_time
            peak_memory   = torch.cuda.max_memory_allocated() / 1024 / 1024
            res = trainer.evaluate()
            start_inf = time.time(); _ = trainer.predict(val_ds); inf_time = time.time() - start_inf

            # ✅ 记录最终门控值
            gate_final = torch.sigmoid(model.fusion_weight).item() if hasattr(model, 'fusion_weight') else 0.0
            row = {
                "Dataset": "SMP2020", "N": N_SAMPLES, "Method": exp['name'], "Seed": SEED,
                "Macro-F1": res['eval_macro_f1'], "Weighted-F1": res['eval_weighted_f1'],
                "Accuracy": res['eval_accuracy'], "Balanced_Acc": res['eval_balanced_acc'],
                "G-Mean": res['eval_g_mean'], "AUC": res['eval_auc'],
                "Gate_Final_σ(λ)": gate_final,
                "Train_Time_Sec": train_runtime, "Inference_Time_Sec": inf_time,
                "Peak_Memory_MB": peak_memory, "Params_M": num_params / 1e6
            }
            for i in range(NUM_LABELS): row[f"F1_Class_{i}"] = res[f"eval_f1_class_{i}"]
            append_to_csv(MAIN_RESULTS_FILE, row)
            del model, trainer; torch.cuda.empty_cache(); gc.collect()
            shutil.rmtree(output_dir, ignore_errors=True)

# ==================== PART B: Sensitivity Analysis ====================
print(f"\n{'='*80}\nPART B: SENSITIVITY (MemSize & TailWeight)\n{'='*80}")
cfg_s     = CONFIGS[2000]
exp_ours  = EXPERIMENTS[0]
train_pool, val_pool = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
val_df = get_custom_dataset(val_pool, {k: 50 for k in range(NUM_LABELS)}, 42)
val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(
    ["input_ids", "attention_mask", "label"]
)

def run_sens(param_name, values):
    for val in values:
        for SEED in RANDOM_SEEDS:
            print(f"\n[Sens] {param_name}={val} | Seed={SEED}")
            set_seed(SEED)
            train_df = get_custom_dataset(train_pool, cfg_s["train"], SEED)
            cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
            class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(
                ["input_ids", "attention_mask", "label"]
            )
            cur_mem  = val if param_name == "MemSize"     else cfg_s["memory_size"]
            cur_tw   = val if param_name == "TailWeight"  else cfg_s["tail_weight"]
            cur_temp = val if param_name == "Temperature" else cfg_s["temperature"]
            cur_lw   = val if param_name == "LossWeight"  else cfg_s["loss_weight"]

            m_cfg = exp_ours.copy(); m_cfg["fusion_init"] = cfg_s["fusion_init"]
            model = UnifiedModel(m_cfg).to(device)

            output_dir    = f"./tmp_smp2020_sensitivity_sens_{param_name}_{val}_{SEED}"
            gate_log_path = f"{GATE_LOG_DIR}/gate_sens_{param_name}_{val}_seed{SEED}.csv"
            gate_cb       = GateMonitorCallback(model, gate_log_path)

            trainer = UnifiedTrainer(
                loss_type="original", class_weights=class_weights_np,
                cls_num_list=[len(train_df[train_df['label'] == i]) for i in range(NUM_LABELS)],
                memory_loss=MemoryBank(128, NUM_LABELS, cur_mem, cur_temp,
                                       TAIL_CLASSES, cur_tw, cfg_s["warmup_steps"], 5).to(device),
                loss_weight=cur_lw, is_peft=True,
                smoothing=cfg_s["smoothing"], use_class_weight=True,
                model=model, train_dataset=train_ds, eval_dataset=val_ds,
                tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                args=make_training_args(output_dir, cfg_s, PEFT_LR * cfg_s["lr_scale"]),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=8), gate_cb]
            )
            trainer.train()
            res        = trainer.evaluate()
            gate_final = torch.sigmoid(model.fusion_weight).item()
            append_to_csv(SENSITIVITY_FILE, {
                "Type": param_name, "Value": val, "Seed": SEED,
                "Macro_F1":    res['eval_macro_f1'],
                "Weighted-F1": res['eval_weighted_f1'],
                "Accuracy":    res['eval_accuracy'],
                "G-Mean":      res['eval_g_mean'],
                "AUC":         res['eval_auc'],
                "Gate_Final_σ(λ)": gate_final
            })
            del model, trainer; torch.cuda.empty_cache(); gc.collect()
            shutil.rmtree(output_dir, ignore_errors=True)

run_sens("MemSize",     SENS_MEM_SIZES)
run_sens("TailWeight",  SENS_TAIL_WEIGHTS)
run_sens("Temperature", SENS_TEMPS)
run_sens("LossWeight",  SENS_LOSS_WEIGHTS)

print(f"\n{'='*80}\nSMP2020 DONE.\n{'='*80}")
print(f"  主实验结果: {MAIN_RESULTS_FILE}")
print(f"  敏感性结果: {SENSITIVITY_FILE}")
print(f"  门控过程日志: {GATE_LOG_DIR}/  (每个run一个CSV，列: step / epoch / gate_sigmoid)")