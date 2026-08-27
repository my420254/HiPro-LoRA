#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as font_manager
import seaborn as sns
import warnings
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ==================== Font ====================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
font_path = os.path.join(BASE_DIR, 'Times_New_Roman.ttf')
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'

# ==================== Output ====================
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

MAIN_RESULT_CSV_PATHS = {
    'SMP2020': [
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'smp2020_main_table2_strict_results.csv'),
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'smp2020_lora_adv_table2_strict_results.csv')
    ],
    'SST-5': [
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'sst5_main_table2_strict_results.csv'),
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'sst5_lora_adv_table2_strict_results.csv')
    ],
    'TweetEval': [
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'tweeteval_main_table2_strict_results.csv'),
        os.path.join(BASE_DIR, 'table2_strict', 'results', 'tweeteval_lora_adv_table2_strict_results.csv')
    ]
}

# ==================== Global Style ====================
plt.rcParams.update({
    'font.size': 36,
    'axes.titlesize': 64,
    'axes.labelsize': 52,
    'xtick.labelsize': 48,
    'ytick.labelsize': 48,
    'legend.fontsize': 32,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 2.0,
    'lines.linewidth': 5.0,
    'lines.markersize': 14,
    'grid.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42
})

# ==================== Colors ====================
C_OURS       = '#F05A28'
C_BLUE_MAIN  = '#1E5FA0'
C_BLUE_LIGHT = '#9CC4E4'
C_GREY_MAIN  = '#7F8C8D'
C_GREY_LIGHT = '#D3D3D3'
C_ADV        = '#2E8B57'
C_DORA       = '#27AE60'
C_LORA_BAL   = '#8E44AD'

HATCH_NO_COT = '//'
HATCH_COT    = '\\\\'
HATCH_OURS   = None

METHOD_COLOR_MAP = {
    'LoRA-Ours':            C_OURS,
    'LoRA-Adv':             C_ADV,
    'DoRA-Balanced':        C_DORA,
    'Full-FineTuning':      C_GREY_MAIN,
    'LoRA-Balanced':        C_LORA_BAL,
    'LoRA-Ablation-NoMem':  C_BLUE_LIGHT,
    'LoRA-Ablation-NoHSP':  C_BLUE_MAIN
}

METHOD_NAME_MAP = {
    'LoRA-Ours':            'HiPro-LoRA',
    'LoRA-Adv':             'LoRA-Adv',
    'DoRA-Balanced':        'DoRA',
    'Full-FineTuning':      'Full-FT',
    'LoRA-Balanced':        'LoRA-Bal',
    'LoRA-Ablation-NoMem':  'w/o TPMB',
    'LoRA-Ablation-NoHSP':  'w/o AHSP'
}

MARKER_MAP = {
    'LoRA-Ours':        's',
    'LoRA-Adv':         '^',
    'DoRA-Balanced':    '*',
    'LoRA-Balanced':    'o',
    'Full-FineTuning':  'X'
}

# ==================== Selected Params (Sensitivity) ====================
SELECTED_PARAMS = {
    "SMP2020":   {"MemSize": 2000, "TailWeight": 3.0, "Temperature": 0.1,  "LossWeight": 0.1},
    "SST-5":     {"MemSize": 1200, "TailWeight": 1.0, "Temperature": 0.5,  "LossWeight": 0.2},
    "TweetEval": {"MemSize": 1500, "TailWeight": 2.0, "Temperature": 0.2,  "LossWeight": 0.06}
}

# ==================== Data Loaders ====================
def load_main_result_csv(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df_raw = pd.read_csv(filepath)
    required = {'Dataset', 'N', 'Method'}
    if not required.issubset(df_raw.columns):
        return pd.DataFrame()

    metric_map = {
        'Macro-F1': 'Macro-F1 (Mean±Std)',
        'Accuracy': 'Accuracy (Mean±Std)',
        'Train_Time_Sec': 'Train_Time_Sec (Mean±Std)',
        'Inference_Time_Sec': 'Inference_Time_Sec (Mean±Std)',
        'Peak_Memory_MB': 'Peak_Memory_MB (Mean±Std)',
        'Params_M': 'Params_M (Mean±Std)'
    }
    for source_col in metric_map:
        if source_col in df_raw.columns:
            df_raw[source_col] = pd.to_numeric(df_raw[source_col], errors='coerce')

    rows = []
    for (dataset, n_val, method), group in df_raw.groupby(['Dataset', 'N', 'Method'], sort=True):
        row = {'Dataset': dataset, 'N': int(n_val), 'Method': method}
        if 'Macro-F1' in group.columns:
            best_idx = group['Macro-F1'].astype(float).idxmax()
            best_row = group.loc[best_idx]
            row['Best (Seed/F1)'] = f"Seed {best_row.get('Seed', '')}: {best_row['Macro-F1']:.4f}"
        for source_col, target_col in metric_map.items():
            if source_col in group.columns:
                row[target_col] = group[source_col].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def load_main_result_data(dataset):
    frames = [load_main_result_csv(path) for path in MAIN_RESULT_CSV_PATHS[dataset]]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sensitivity_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "## Direct Plot Data":
            start = i + 1
            break

    data_lines = [l for l in lines[start:] if l.strip().startswith('|') and '---' not in l]
    md_content = '\n'.join(data_lines)
    df = pd.read_csv(io.StringIO(md_content), sep='|')
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.replace('*', '').str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('*', '').str.strip()
    df['Value'] = pd.to_numeric(df['Value'])
    seed_cols = [c for c in df.columns if 'Seed' in c]
    df_melted = df.melt(
        id_vars=['Dataset', 'Type', 'Value'],
        value_vars=seed_cols,
        var_name='Seed',
        value_name='Macro_F1'
    )
    df_melted['Macro_F1'] = pd.to_numeric(df_melted['Macro_F1']) * 100
    return df_melted


# ==================== Figure: LLM Comparison ====================
def plot_llm_comparison(df_smp, df_sst, df_tweet):
    strict_csv_path = os.path.join(BASE_DIR, 'llm_baselines', 'few_shot_results', 'llm_fewshot_results.csv')
    if os.path.exists(strict_csv_path):
        df_llm = pd.read_csv(strict_csv_path)
        if 'Protocol' in df_llm.columns:
            df_llm = df_llm[df_llm['Protocol'].astype(str) == 'balanced-held-out-test'].copy()
        df_llm['Macro-F1'] = pd.to_numeric(df_llm['Macro-F1'], errors='coerce')

        datasets = ['SMP2020', 'SST-5', 'TweetEval']
        def method_to_shot(method):
            method = str(method)
            if 'Zero-Shot' in method:
                return 0
            for shot in (1, 3, 5):
                if f'{shot}-Shot' in method:
                    return shot
            return np.nan

        def method_to_prompt(method):
            method = str(method)
            if 'No CoT' in method:
                return 'No CoT'
            if 'CoT' in method:
                return 'CoT'
            return 'No CoT'

        def model_to_short(model):
            model = str(model)
            if 'Qwen' in model:
                return 'Qwen'
            if 'Llama' in model:
                return 'Llama'
            return model

        def best_ours_for_dataset(dataset):
            csv_frames = [pd.read_csv(path) for path in MAIN_RESULT_CSV_PATHS.get(dataset, []) if os.path.exists(path)]
            if not csv_frames:
                return None, None
            df_main = pd.concat(csv_frames, ignore_index=True)
            rows = df_main[df_main['Method'].astype(str) == 'LoRA-Ours'].copy()
            if rows.empty:
                return None, None
            rows['N'] = pd.to_numeric(rows['N'], errors='coerce')
            rows['Macro-F1'] = pd.to_numeric(rows['Macro-F1'], errors='coerce')
            grouped = rows.dropna(subset=['N', 'Macro-F1']).groupby('N', as_index=False)['Macro-F1'].mean()
            if grouped.empty:
                return None, None
            best = grouped.sort_values('Macro-F1', ascending=False).iloc[0]
            return float(best['Macro-F1']) * 100, int(best['N'])

        df_llm['Shot'] = df_llm['Method'].map(method_to_shot)
        df_llm['Prompt'] = df_llm['Method'].map(method_to_prompt)
        df_llm['ModelShort'] = df_llm['Model'].map(model_to_short)
        df_llm['MacroF1Pct'] = df_llm['Macro-F1'] * 100
        df_llm = df_llm.dropna(subset=['Shot', 'MacroF1Pct'])

        series_specs = [
            ('Qwen',  'No CoT', C_BLUE_MAIN,  'o', '-'),
            ('Qwen',  'CoT',    C_BLUE_MAIN,  's', '--'),
            ('Llama', 'No CoT', C_ADV,        '^', '-'),
            ('Llama', 'CoT',    C_ADV,        'D', '--'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(34, 11), sharey=True)
        for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
            subset = df_llm[df_llm['Dataset'].astype(str) == dataset]
            for model, prompt, color, marker, linestyle in series_specs:
                rows = subset[(subset['ModelShort'] == model) & (subset['Prompt'] == prompt)].sort_values('Shot')
                if rows.empty:
                    continue
                ax.plot(
                    rows['Shot'],
                    rows['MacroF1Pct'],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=4,
                    markersize=12,
                    markeredgecolor='black',
                    markeredgewidth=1.2,
                    alpha=0.95,
                )

            ours_pct, ours_n = best_ours_for_dataset(dataset)
            if ours_pct is not None:
                ax.axhline(ours_pct, color=C_OURS, linewidth=5, linestyle='-', zorder=1)
                ax.text(
                    5.18,
                    ours_pct,
                    f'{ours_pct:.1f}%\nN={ours_n}',
                    color=C_OURS,
                    fontsize=23,
                    fontweight='bold',
                    va='center',
                    ha='left',
                )

            if not subset.empty:
                best_llm = subset.sort_values('MacroF1Pct', ascending=False).iloc[0]
                best_x = float(best_llm['Shot'])
                best_y = float(best_llm['MacroF1Pct'])
                offset_y = 3.2 if dataset == 'SMP2020' else (-6.2 if best_y >= 63 else 5.0)
                offset_x = 0.35 if best_x <= 3 else -1.15
                ax.annotate(
                    f'Best LLM\n{best_y:.1f}%',
                    xy=(best_x, best_y),
                    xytext=(best_x + offset_x, best_y + offset_y),
                    fontsize=22,
                    fontweight='bold',
                    color='#222222',
                    ha='left' if offset_x > 0 else 'right',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#555555', lw=1.2, alpha=0.92),
                    arrowprops=dict(arrowstyle='->', lw=1.4, color='#333333', shrinkA=4, shrinkB=4),
                    zorder=20,
                )

            ax.set_title(f'({chr(65 + idx)}) {dataset}', fontweight='bold', pad=24, fontsize=44)
            ax.set_xlabel('Shot Count', fontweight='bold', fontsize=34)
            if idx == 0:
                ax.set_ylabel('Macro-F1 Score (%)', fontweight='bold', fontsize=34)
            ax.set_xticks([0, 1, 3, 5])
            ax.set_xlim(-0.25, 5.65)
            ax.set_ylim(35, 80)
            ax.tick_params(axis='both', labelsize=28, width=2, length=7)
            ax.grid(axis='y', linestyle='--', alpha=0.28, linewidth=1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

        legend_handles = [
            mlines.Line2D([], [], color=C_BLUE_MAIN, marker='o', linestyle='-', linewidth=4,
                          markersize=12, markeredgecolor='black', label='Qwen No CoT'),
            mlines.Line2D([], [], color=C_BLUE_MAIN, marker='s', linestyle='--', linewidth=4,
                          markersize=12, markeredgecolor='black', label='Qwen CoT'),
            mlines.Line2D([], [], color=C_ADV, marker='^', linestyle='-', linewidth=4,
                          markersize=12, markeredgecolor='black', label='Llama No CoT'),
            mlines.Line2D([], [], color=C_ADV, marker='D', linestyle='--', linewidth=4,
                          markersize=12, markeredgecolor='black', label='Llama CoT'),
            mlines.Line2D([], [], color=C_OURS, linestyle='-', linewidth=5, label='HiPro-LoRA best'),
        ]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                   ncol=5, frameon=False, fontsize=28, columnspacing=1.4, handlelength=2.5)
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig(os.path.join(FIG_DIR, 'figure4_llm.pdf'))
        plt.close()
        print("figure4_llm.pdf saved from full strict LLM CSV.")
        return

    print(f"Strict LLM CSV not found: {strict_csv_path}. Skip figure4_llm.pdf.")


# ==================== Figure: Efficiency ====================
def plot_efficiency(df_smp, df_sst, df_tweet):
    tasks      = [('SMP2020', df_smp, 1000), ('TweetEval', df_tweet, 1000), ('SST-5', df_sst, 1150)]
    main_m     = ['LoRA-Ours', 'LoRA-Adv', 'DoRA-Balanced', 'LoRA-Balanced', 'Full-FineTuning']
    fig, axes  = plt.subplots(1, 3, figsize=(32, 15))
    legend_handles = []
    method_seen    = set()

    for i, (name, df, n_val) in enumerate(tasks):
        ax = axes[i]
        if df.empty:
            continue
        subset = df[(df['N'] == n_val) & (df['Method'].isin(main_m))].copy()
        for m in main_m:
            row = subset[subset['Method'] == m]
            if row.empty:
                continue
            row = row.iloc[0]
            t   = row['Train_Time_Sec (Mean±Std)']
            mem = row['Peak_Memory_MB (Mean±Std)']
            f1  = row['Macro-F1 (Mean±Std)']
            if pd.isna(t) or pd.isna(mem):
                continue
            col    = METHOD_COLOR_MAP.get(m, '#333333')
            marker = MARKER_MAP.get(m, 'o')
            ax.scatter(t, mem, s=(f1 - 0.3) * 12000, color=col, alpha=0.7,
                       edgecolor='black', linewidth=2, marker=marker)
            if i == 0 and m not in method_seen:
                legend_handles.append(
                    mlines.Line2D([], [], color=col, marker=marker, linestyle='None',
                                  markersize=60, markeredgecolor='black', markeredgewidth=3,
                                  label=METHOD_NAME_MAP.get(m, m))
                )
                method_seen.add(m)

        ax.tick_params(axis='both', which='major', labelsize=52, width=3, length=10)
        ax.set_xlabel('Training Time (s)', fontweight='bold', fontsize=55)
        if i == 0:
            ax.set_ylabel('Peak Memory (MB)', fontweight='bold', fontsize=48)
        ax.set_title(f'({chr(65 + i)}) {name}', weight='bold', pad=30, fontsize=64)
        ax.grid(True, ls='--', alpha=0.3, linewidth=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.set_xlim(subset['Train_Time_Sec (Mean±Std)'].min() * 0.7,
                    subset['Train_Time_Sec (Mean±Std)'].max() * 1.3)
        ax.set_ylim(subset['Peak_Memory_MB (Mean±Std)'].min() * 0.8,
                    subset['Peak_Memory_MB (Mean±Std)'].max() * 1.3)

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, prop={'size': 50})
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(os.path.join(FIG_DIR, 'figure5_efficiency.pdf'))
    plt.close()
    print("figure5_efficiency.pdf saved.")


# ==================== Figure: Sensitivity ====================
def plot_sensitivity():
    sens_path = os.path.join(BASE_DIR, 'Figure5_sensitivity_data.md')
    full_df   = load_sensitivity_data(sens_path)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes      = axes.flatten()

    plot_configs = [
        {"type": "MemSize",     "ax_idx": 0, "title": "(a) Effect of Memory Size ($M$)",      "xlabel": "Memory Size"},
        {"type": "TailWeight",  "ax_idx": 1, "title": "(b) Effect of Tail Weight ($\\gamma$)", "xlabel": "Tail Weight"},
        {"type": "Temperature", "ax_idx": 2, "title": "(c) Effect of Temperature ($\\tau$)",   "xlabel": "Temperature"},
        {"type": "LossWeight",  "ax_idx": 3, "title": "(d) Effect of Loss Weight ($\\beta$)",  "xlabel": "Loss Weight"}
    ]

    datasets = full_df['Dataset'].unique()
    palette  = sns.color_palette("Set1", len(datasets))

    for cfg in plot_configs:
        ax     = axes[cfg["ax_idx"]]
        subset = full_df[full_df['Type'] == cfg["type"]]

        sns.lineplot(data=subset, x="Value", y="Macro_F1", hue="Dataset",
                     style="Dataset", markers=True, ax=ax,
                     palette=palette, errorbar='sd', legend=False)

        for i, ds in enumerate(datasets):
            target_val = SELECTED_PARAMS.get(ds, {}).get(cfg["type"])
            if target_val is not None:
                ds_data   = subset[subset['Dataset'] == ds]
                mean_line = ds_data.groupby('Value')['Macro_F1'].mean().sort_index()
                if not mean_line.empty:
                    interp_y = np.interp(target_val, mean_line.index, mean_line.values)
                    ax.scatter(target_val, interp_y,
                               marker='*', s=600,
                               color='gold', edgecolor='black',
                               linewidth=1.5, zorder=10)

        ax.set_title(cfg["title"], fontweight='bold', pad=25, fontsize=32)
        ax.set_xlabel(cfg["xlabel"], fontweight='bold', labelpad=15, fontsize=28)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=2.0)
        ax.margins(y=0.15)
        if cfg["ax_idx"] % 2 == 0:
            ax.set_ylabel("Macro-F1 (%)", fontweight='bold', labelpad=15, fontsize=28)
        else:
            ax.set_ylabel("")

    plt.subplots_adjust(hspace=0.4, wspace=0.15, bottom=0.15)

    from matplotlib.lines import Line2D
    custom_handles = []
    custom_labels  = []
    for i, ds in enumerate(datasets):
        custom_handles.append(
            Line2D([0], [0], color=palette[i], linestyle='-', linewidth=8, marker='o', markersize=20)
        )
        custom_labels.append(ds)
    custom_handles.append(
        Line2D([0], [0], color='w', marker='*', markerfacecolor='gold',
               markeredgecolor='black', markersize=36, markeredgewidth=2.0)
    )
    custom_labels.append("Reference Config")

    fig.legend(custom_handles, custom_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.06),
               ncol=len(custom_handles), frameon=False,
               fontsize=36, columnspacing=3.0, handletextpad=1.0)

    plt.savefig(os.path.join(FIG_DIR, 'figure6_sens.pdf'), bbox_inches='tight')
    plt.close()
    print("figure6_sens.pdf saved.")


# ==================== Main ====================
if __name__ == "__main__":
    df_smp   = load_main_result_data('SMP2020')
    df_sst   = load_main_result_data('SST-5')
    df_tweet = load_main_result_data('TweetEval')

    plot_llm_comparison(df_smp, df_sst, df_tweet)
    plot_efficiency(df_smp, df_sst, df_tweet)
    plot_sensitivity()
