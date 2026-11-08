"""
Huan luyen & Fine-tuning mo hinh cho IEEE-CIS Fraud Detection.

Vai tro  : Chay cac mo hinh, fine-tune
Nhiem vu : Xu ly mat can bang (class weight, SMOTE, undersampling)
           + Logistic Regression, Random Forest, XGBoost, fine-tuning

Pipeline:
  1. Load du lieu sach tu preprocess.py
  2. Chay 9 thi nghiem: 3 mo hinh x 3 chien luoc mat can bang
  3. Fine-tune hyperparameters (tren tap val) cho tung mo hinh
  4. Danh gia cuoi cung tren tap test
  5. Xuat bang so sanh, bieu do ROC, confusion matrix, feature importance

Dau vao : data/clean/{train,val,test}.parquet          (cho RF, XGBoost)
          data/clean/{train,val,test}_scaled.parquet   (cho LR)
Dau ra  : models/*.pkl
          reports/models/*.png, *.csv

Cai them: pip install imbalanced-learn  (cho SMOTE)

Luu y: SMOTE va undersampling CHI ap dung tren tap TRAIN.
"""

from pathlib import Path
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# =====================================================================
# CAU HINH
# =====================================================================
DATA = Path("data/clean")
MODEL_DIR = Path("models")
REPORT = Path("reports/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
LABEL = "isFraud"
SEED = 42

# =====================================================================
# HAM HO TRO
# =====================================================================

def load_data():
    """Load du lieu sach. Tra ve dict {split: (X, y)} cho ca 2 phien ban."""
    data = {}
    for split in ["train", "val", "test"]:
        # Phien ban KHONG scale (cho cay: RF, XGBoost)
        df = pd.read_parquet(DATA / f"{split}.parquet")
        y = df.pop(LABEL).values
        X = df.values
        cols = df.columns.tolist()
        data[f"{split}_tree"] = (X, y)

        # Phien ban SCALED (cho LR)
        df_s = pd.read_parquet(DATA / f"{split}_scaled.parquet")
        y_s = df_s.pop(LABEL).values
        X_s = df_s.values
        cols_s = df_s.columns.tolist()
        data[f"{split}_lr"] = (X_s, y_s)

    data["feature_names_tree"] = cols
    data["feature_names_lr"] = cols_s
    return data


def apply_imbalance_strategy(X_train, y_train, strategy):
    """
    Ap dung chien luoc xu ly mat can bang CHI tren tap train.
    Tra ve (X_resampled, y_resampled).
    """
    if strategy == "class_weight":
        # Khong thay doi du lieu; truyen class_weight vao mo hinh
        return X_train, y_train

    elif strategy == "smote":
        sm = SMOTE(random_state=SEED, n_jobs=-1)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"    SMOTE: {len(y_train):,} -> {len(y_res):,} mau "
              f"(fraud: {y_train.sum():,} -> {y_res.sum():,})")
        return X_res, y_res

    elif strategy == "undersampling":
        rus = RandomUnderSampler(random_state=SEED)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print(f"    Undersampling: {len(y_train):,} -> {len(y_res):,} mau "
              f"(fraud: {y_train.sum():,} -> {y_res.sum():,})")
        return X_res, y_res

    else:  # "none" - baseline
        return X_train, y_train


def get_model(model_name, strategy, params=None):
    """Tao mo hinh voi hyperparameters cho truoc."""
    use_cw = (strategy == "class_weight")

    if model_name == "LogisticRegression":
        default = dict(
            max_iter=500,
            solver="lbfgs",          # nhanh va on dinh hon saga
            class_weight="balanced" if use_cw else None,
            random_state=SEED,
            n_jobs=-1,
        )
        if params:
            default.update(params)
        return LogisticRegression(**default)

    elif model_name == "RandomForest":
        default = dict(
            n_estimators=300,
            max_depth=18,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced" if use_cw else None,
            random_state=SEED,
            n_jobs=-1,
        )
        if params:
            default.update(params)
        return RandomForestClassifier(**default)

    elif model_name == "XGBoost":
        # Tinh scale_pos_weight tu du lieu goc
        default = dict(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
        )
        if use_cw:
            # scale_pos_weight = so mau am / so mau duong
            default["scale_pos_weight"] = 28.56  # ~342336/11988
        if params:
            default.update(params)
        return XGBClassifier(**default)


def evaluate(model, X, y, label=""):
    """Danh gia mo hinh, tra ve dict metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred)

    if label:
        print(f"    {label}: AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f} | "
              f"F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f}")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


# =====================================================================
# BUOC 1: CHAY 9 THI NGHIEM (3 mo hinh x 3 chien luoc)
# =====================================================================

def run_baseline_experiments(data):
    """Chay 9 to hop: 3 mo hinh x 3 chien luoc xu ly mat can bang."""
    results = []

    models = ["LogisticRegression", "RandomForest", "XGBoost"]
    strategies = ["class_weight", "smote", "undersampling"]

    for model_name in models:
        # LR dung du lieu scaled; cay dung du lieu khong scale
        suffix = "lr" if model_name == "LogisticRegression" else "tree"
        X_train, y_train = data[f"train_{suffix}"]
        X_val, y_val = data[f"val_{suffix}"]

        for strategy in strategies:
            exp_name = f"{model_name}__{strategy}"
            print(f"\n{'='*60}")
            print(f"[THI NGHIEM] {exp_name}")
            print(f"{'='*60}")

            # Xu ly mat can bang (chi tren train)
            X_tr, y_tr = apply_imbalance_strategy(X_train, y_train, strategy)

            # Tao va huan luyen mo hinh
            model = get_model(model_name, strategy)
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = time.time() - t0
            print(f"    Thoi gian huan luyen: {train_time:.1f}s")

            # Danh gia tren VAL
            metrics = evaluate(model, X_val, y_val, label="Val")

            results.append({
                "model": model_name,
                "strategy": strategy,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "train_time_s": round(train_time, 1),
                "train_samples": len(y_tr),
            })

            # Luu model
            with open(MODEL_DIR / f"{exp_name}.pkl", "wb") as f:
                pickle.dump(model, f)

    return pd.DataFrame(results)


# =====================================================================
# BUOC 2: FINE-TUNING HYPERPARAMETERS
# =====================================================================

# Khong gian tim kiem cho tung mo hinh
PARAM_GRID = {
    "LogisticRegression": [
        {"C": 0.01,  "penalty": "l2"},
        {"C": 0.1,   "penalty": "l2"},
        {"C": 1.0,   "penalty": "l2"},
        {"C": 10.0,  "penalty": "l2"},
        {"C": 0.1,   "penalty": "l1", "solver": "saga", "max_iter": 300},
        {"C": 1.0,   "penalty": "l1", "solver": "saga", "max_iter": 300},
    ],
    "RandomForest": [
        {"n_estimators": 200, "max_depth": 12, "min_samples_split": 20, "min_samples_leaf": 8},
        {"n_estimators": 300, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_estimators": 300, "max_depth": 20, "min_samples_split": 5,  "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 18, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_estimators": 400, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4},
    ],
    "XGBoost": [
        {"n_estimators": 300, "max_depth": 5,  "learning_rate": 0.1,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0,  "reg_lambda": 1.0},
        {"n_estimators": 500, "max_depth": 7,  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,  "reg_lambda": 1.0},
        {"n_estimators": 700, "max_depth": 7,  "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5,  "reg_lambda": 2.0},
        {"n_estimators": 500, "max_depth": 9,  "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.01, "reg_lambda": 0.5},
        {"n_estimators": 800, "max_depth": 6,  "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 1.0,  "reg_lambda": 3.0},
        {"n_estimators": 1000,"max_depth": 8,  "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,  "reg_lambda": 1.0},
    ],
}


def fine_tune(data, baseline_results):
    """
    Fine-tune: voi moi mo hinh, chon chien luoc tot nhat (theo AUC-ROC tren val),
    roi thu nhieu bo hyperparameters.
    Toi uu: cache du lieu SMOTE/undersampling de khong phai tinh lai.
    """
    tuning_results = []
    # Cache du lieu da resample de khong phai goi SMOTE nhieu lan
    resampled_cache = {}

    for model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        # Tim chien luoc tot nhat cho mo hinh nay tu baseline
        mask = baseline_results["model"] == model_name
        best_row = baseline_results.loc[mask].sort_values("auc_roc", ascending=False).iloc[0]
        best_strategy = best_row["strategy"]

        print(f"\n{'#'*60}")
        print(f"[FINE-TUNE] {model_name} | Chien luoc tot nhat: {best_strategy} "
              f"(baseline AUC-ROC={best_row['auc_roc']:.4f})")
        print(f"{'#'*60}")

        suffix = "lr" if model_name == "LogisticRegression" else "tree"
        X_train, y_train = data[f"train_{suffix}"]
        X_val, y_val = data[f"val_{suffix}"]

        # Ap dung chien luoc mat can bang (cache lai de khong tinh lai)
        cache_key = f"{suffix}__{best_strategy}"
        if cache_key in resampled_cache:
            X_tr, y_tr = resampled_cache[cache_key]
            print(f"    (dung du lieu da cache: {len(y_tr):,} mau)")
        else:
            X_tr, y_tr = apply_imbalance_strategy(X_train, y_train, best_strategy)
            resampled_cache[cache_key] = (X_tr, y_tr)

        best_auc = 0
        best_params = None
        best_model = None

        for i, params in enumerate(PARAM_GRID[model_name]):
            print(f"\n  [{i+1}/{len(PARAM_GRID[model_name])}] {params}")
            model = get_model(model_name, best_strategy, params)
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = time.time() - t0
            print(f"    Train time: {train_time:.1f}s")

            metrics = evaluate(model, X_val, y_val, label=f"  Val")

            tuning_results.append({
                "model": model_name,
                "strategy": best_strategy,
                "params": str(params),
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1_macro": metrics["f1_macro"],
                "train_time_s": round(train_time, 1),
            })

            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_params = params
                best_model = model

        print(f"\n  >>> BEST: AUC-ROC={best_auc:.4f} | params={best_params}")

        # Luu mo hinh tot nhat sau fine-tune
        fname = f"{model_name}__tuned__{best_strategy}.pkl"
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(best_model, f)

    return pd.DataFrame(tuning_results)


# =====================================================================
# BUOC 3: DANH GIA CUOI CUNG TREN TAP TEST
# =====================================================================

def final_evaluation(data):
    """Load cac mo hinh tuned, danh gia tren TEST, xuat bao cao."""
    tuned_files = sorted(MODEL_DIR.glob("*__tuned__*.pkl"))
    if not tuned_files:
        print("Khong tim thay mo hinh tuned!")
        return

    final_results = []
    all_curves = {}  # de ve ROC

    for fpath in tuned_files:
        model_name = fpath.stem.split("__tuned__")[0]
        strategy = fpath.stem.split("__tuned__")[1]

        with open(fpath, "rb") as f:
            model = pickle.load(f)

        suffix = "lr" if model_name == "LogisticRegression" else "tree"
        X_test, y_test = data[f"test_{suffix}"]

        metrics = evaluate(model, X_test, y_test, label=f"TEST {model_name}")
        final_results.append({
            "model": model_name,
            "strategy": strategy,
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        })

        all_curves[model_name] = {
            "y_test": y_test,
            "y_prob": metrics["y_prob"],
            "cm": metrics["confusion_matrix"],
        }

    # --- Bang ket qua cuoi cung ---
    df_final = pd.DataFrame(final_results).sort_values("auc_roc", ascending=False)
    df_final.to_csv(REPORT / "final_test_results.csv", index=False)
    print(f"\n{'='*60}")
    print("KET QUA CUOI CUNG TREN TAP TEST")
    print(f"{'='*60}")
    print(df_final.to_string(index=False))

    # --- Ve ROC curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"LogisticRegression": "#4c78a8", "RandomForest": "#72b7b2", "XGBoost": "#e45756"}

    for name, info in all_curves.items():
        fpr, tpr, _ = roc_curve(info["y_test"], info["y_prob"])
        auc_val = roc_auc_score(info["y_test"], info["y_prob"])
        axes[0].plot(fpr, tpr, color=colors.get(name, "gray"),
                     label=f"{name} (AUC={auc_val:.4f})", linewidth=2)

        prec, rec, _ = precision_recall_curve(info["y_test"], info["y_prob"])
        ap = average_precision_score(info["y_test"], info["y_prob"])
        axes[1].plot(rec, prec, color=colors.get(name, "gray"),
                     label=f"{name} (AP={ap:.4f})", linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test Set)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(REPORT / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, len(all_curves), figsize=(6 * len(all_curves), 5))
    if len(all_curves) == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, all_curves.items()):
        sns.heatmap(info["cm"], annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Binh thuong", "Gian lan"],
                    yticklabels=["Binh thuong", "Gian lan"])
        ax.set_title(f"{name}")
        ax.set_ylabel("Thuc te")
        ax.set_xlabel("Du doan")
    fig.suptitle("Confusion Matrix (Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Feature importance (RF va XGBoost) ---
    plot_feature_importance(data)

    return df_final


def plot_feature_importance(data):
    """Ve feature importance cho Random Forest va XGBoost (mo hinh tuned)."""
    feature_names = data["feature_names_tree"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, model_name in enumerate(["RandomForest", "XGBoost"]):
        tuned_files = list(MODEL_DIR.glob(f"{model_name}__tuned__*.pkl"))
        if not tuned_files:
            continue
        with open(tuned_files[0], "rb") as f:
            model = pickle.load(f)

        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-25:]  # top 25
        axes[idx].barh(
            [feature_names[i] for i in top_idx],
            imp[top_idx],
            color="#4c78a8" if model_name == "RandomForest" else "#e45756",
        )
        axes[idx].set_title(f"Top 25 Feature Importance - {model_name}")
        axes[idx].set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(REPORT / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Feature importance saved -> {REPORT / 'feature_importance.png'}")


# =====================================================================
# BUOC 4: TONG HOP BAO CAO
# =====================================================================

def generate_summary(baseline_df, tuning_df, final_df):
    """Tao bao cao tong hop."""
    # Luu tat ca ket qua
    baseline_df.to_csv(REPORT / "baseline_results.csv", index=False)
    if tuning_df is not None:
        tuning_df.to_csv(REPORT / "tuning_results.csv", index=False)

    # --- Bieu do so sanh baseline ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_model = {"LogisticRegression": "#4c78a8", "RandomForest": "#72b7b2", "XGBoost": "#e45756"}

    # AUC-ROC
    for model_name in baseline_df["model"].unique():
        mask = baseline_df["model"] == model_name
        sub = baseline_df[mask]
        x = range(len(sub))
        bars = axes[0].bar(
            [i + list(baseline_df["model"].unique()).index(model_name) * 0.25
             for i in range(len(sub))],
            sub["auc_roc"],
            width=0.25,
            label=model_name,
            color=colors_model.get(model_name, "gray"),
        )

    # Ve lai dep hon: grouped bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    strategies = baseline_df["strategy"].unique()
    models = baseline_df["model"].unique()
    x = np.arange(len(strategies))
    width = 0.25
    for i, model_name in enumerate(models):
        vals = []
        for s in strategies:
            row = baseline_df[(baseline_df["model"] == model_name) & (baseline_df["strategy"] == s)]
            vals.append(row["auc_roc"].values[0] if len(row) > 0 else 0)
        bars = ax2.bar(x + i * width, vals, width,
                       label=model_name, color=colors_model.get(model_name, "gray"))
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Chien luoc xu ly mat can bang")
    ax2.set_ylabel("AUC-ROC (Val)")
    ax2.set_title("So sanh 9 thi nghiem: AUC-ROC tren tap Val")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.set_ylim(bottom=max(0, baseline_df["auc_roc"].min() - 0.05))
    fig2.tight_layout()
    fig2.savefig(REPORT / "baseline_comparison.png", dpi=150)
    plt.close(fig2)
    plt.close(fig)

    # --- Bieu do fine-tuning ---
    if tuning_df is not None and len(tuning_df) > 0:
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        for idx, model_name in enumerate(["LogisticRegression", "RandomForest", "XGBoost"]):
            sub = tuning_df[tuning_df["model"] == model_name].reset_index(drop=True)
            if sub.empty:
                continue
            axes3[idx].bar(range(len(sub)), sub["auc_roc"],
                           color=colors_model.get(model_name, "gray"))
            best_idx = sub["auc_roc"].idxmax()
            axes3[idx].bar(best_idx, sub.loc[best_idx, "auc_roc"],
                           color="gold", edgecolor="black", linewidth=2)
            axes3[idx].set_title(f"{model_name} Fine-tuning")
            axes3[idx].set_xlabel("Param config #")
            axes3[idx].set_ylabel("AUC-ROC (Val)")
            axes3[idx].set_ylim(bottom=max(0, sub["auc_roc"].min() - 0.01))
        fig3.suptitle("Fine-tuning: AUC-ROC theo bo hyperparameters", fontsize=14)
        fig3.tight_layout()
        fig3.savefig(REPORT / "finetuning_comparison.png", dpi=150)
        plt.close(fig3)

    print(f"\n{'='*60}")
    print(f"TAT CA KET QUA DA LUU VAO: {REPORT.resolve()}")
    print(f"MO HINH DA LUU VAO:        {MODEL_DIR.resolve()}")
    print(f"{'='*60}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IEEE-CIS FRAUD DETECTION - HUAN LUYEN & FINE-TUNE")
    print("=" * 60)

    # 1. Load du lieu
    print("\n[1] LOAD DU LIEU...")
    data = load_data()
    X_tr, y_tr = data["train_tree"]
    print(f"    Train: {X_tr.shape[0]:,} mau, {X_tr.shape[1]} features")
    print(f"    Fraud rate: {y_tr.mean():.3%}")

    # Tinh scale_pos_weight chinh xac cho XGBoost
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos
    print(f"    scale_pos_weight = {neg:,}/{pos:,} = {spw:.2f}")

    # 2. Chay 9 thi nghiem baseline
    print("\n[2] CHAY 9 THI NGHIEM BASELINE...")
    baseline_df = run_baseline_experiments(data)
    print("\n--- TONG HOP BASELINE (sap xep theo AUC-ROC) ---")
    print(baseline_df.sort_values("auc_roc", ascending=False).to_string(index=False))

    # 3. Fine-tune
    print("\n[3] FINE-TUNING HYPERPARAMETERS...")
    tuning_df = fine_tune(data, baseline_df)
    print("\n--- TONG HOP FINE-TUNING (sap xep theo AUC-ROC) ---")
    print(tuning_df.sort_values("auc_roc", ascending=False).to_string(index=False))

    # 4. Danh gia cuoi cung tren test
    print("\n[4] DANH GIA CUOI CUNG TREN TAP TEST...")
    final_df = final_evaluation(data)

    # 5. Tong hop bao cao
    print("\n[5] TAO BAO CAO...")
    generate_summary(baseline_df, tuning_df, final_df)

    print("\nHOAN TAT!")

