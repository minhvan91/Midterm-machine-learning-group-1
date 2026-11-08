"""
EDA (phan tich kham pha du lieu) cho IEEE-CIS Fraud Detection.

Dau vao : data/processed/{train,val,test}.parquet   (tu prepare_data.py)
Dau ra  : reports/eda/  (hinh .png + bang .csv + summary.txt) de dan vao bao cao

Cai them: pip install matplotlib
Quy tac : EDA chi nhin tren tap TRAIN (tru bieu do theo thoi gian dung ca 3 tap),
          de khong "nhin trom" tap test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/processed")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)
LABEL = "isFraud"
summary = []


def log(msg=""):
    print(msg)
    summary.append(str(msg))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


df = pd.read_parquet(DATA / "train.parquet")
log(f"Train: {df.shape[0]:,} mau, {df.shape[1]} cot")

# ---------------------------------------------------------------- 1. Mat can bang
counts = df[LABEL].value_counts().sort_index()
rate = df[LABEL].mean()
log(f"\n[1] Nhan: binh thuong={counts[0]:,} | gian lan={counts[1]:,} | ty le gian lan={rate:.3%}")
log(f"    Mo hinh doan 'tat ca binh thuong' dat accuracy = {1 - rate:.3%} (vo dung)")
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(["Binh thuong", "Gian lan"], counts.values, color=["#4c78a8", "#e45756"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_title("Phan phoi nhan (train)")
save(fig, "01_class_balance.png")

# ---------------------------------------------------------------- 2. Gia tri thieu
miss = df.isna().mean().sort_values(ascending=False)
miss.to_csv(OUT / "02_missing_by_column.csv", header=["missing_ratio"])
log(f"\n[2] So cot thieu >50%: {(miss > 0.5).sum()} | >90%: {(miss > 0.9).sum()} | co thieu: {(miss > 0).sum()}")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(miss.values, bins=20, color="#4c78a8")
axes[0].set_xlabel("Ty le thieu cua cot")
axes[0].set_ylabel("So cot")
axes[0].set_title("Phan phoi ty le thieu")
top = miss.head(25)[::-1]
axes[1].barh(top.index, top.values, color="#e45756")
axes[1].set_title("25 cot thieu nhieu nhat")
save(fig, "02_missing.png")

# ---------------------------------------------------------------- 3. So tien giao dich
amt = df.groupby(LABEL)["TransactionAmt"].describe()
amt.to_csv(OUT / "03_amount_stats_by_class.csv")
log("\n[3] TransactionAmt theo nhan:")
log(amt.round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for lab, color, name in [(0, "#4c78a8", "Binh thuong"), (1, "#e45756", "Gian lan")]:
    x = df.loc[df[LABEL] == lab, "TransactionAmt"]
    axes[0].hist(x.clip(upper=x.quantile(0.99)), bins=50, alpha=0.6, density=True, color=color, label=name)
    axes[1].hist(np.log1p(x), bins=50, alpha=0.6, density=True, color=color, label=name)
axes[0].set_title("TransactionAmt (cat o phan vi 99%)")
axes[1].set_title("log1p(TransactionAmt)")
axes[0].legend()
save(fig, "03_amount.png")

# ---------------------------------------------------------------- 4. Theo thoi gian
parts = []
for name in ["train", "val", "test"]:
    p = pd.read_parquet(DATA / f"{name}.parquet", columns=["TransactionDT", LABEL])
    p["split"] = name
    parts.append(p)
t = pd.concat(parts)
t["day"] = t["TransactionDT"] // 86400
daily = t.groupby("day")[LABEL].mean()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily.index, daily.values, color="#4c78a8")
for name, color in [("val", "orange"), ("test", "red")]:
    ax.axvline(t.loc[t["split"] == name, "day"].min(), color=color, ls="--", label=f"bat dau {name}")
ax.set_xlabel("Ngay (tu moc dau tien)")
ax.set_ylabel("Ty le gian lan")
ax.set_title("Ty le gian lan theo ngay (dung de bien luan cho viec chia theo thoi gian)")
ax.legend()
save(fig, "04a_fraud_rate_by_day.png")
log("\n[4] Ty le gian lan theo tap: " + ", ".join(
    f"{k}={v:.3%}" for k, v in t.groupby("split")[LABEL].mean().reindex(["train", "val", "test"]).items()))
del t, parts

df["hour"] = (df["TransactionDT"] // 3600) % 24
hourly = df.groupby("hour")[LABEL].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(hourly.index, hourly["mean"], color="#e45756")
ax.axhline(rate, color="black", ls="--", label="trung binh")
ax.set_xlabel("Gio trong ngay (xap xi, tu TransactionDT)")
ax.set_ylabel("Ty le gian lan")
ax.legend()
save(fig, "04b_fraud_rate_by_hour.png")

# ---------------------------------------------------------------- 5. Bien phan loai
log("\n[5] Ty le gian lan theo nhom (chi nhom >= 200 giao dich):")
for col in ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]:
    if col not in df.columns:
        continue
    g = df.groupby(col, dropna=False)[LABEL].agg(["mean", "count"])
    g = g[g["count"] >= 200].sort_values("count", ascending=False).head(15)
    if g.empty:
        continue
    g.to_csv(OUT / f"05_fraud_by_{col}.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [str(i) for i in g.index]
    ax.bar(labels, g["mean"], color="#e45756")
    ax.axhline(rate, color="black", ls="--")
    ax.set_title(f"Ty le gian lan theo {col}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, f"05_fraud_by_{col}.png")
    log(f"    {col}: cao nhat = {g['mean'].idxmax()} ({g['mean'].max():.2%})")

# ---------------------------------------------------------------- 6. So gia tri khac nhau cua cot chu
cat_cols = df.select_dtypes(exclude="number").columns
card = df[cat_cols].nunique().sort_values(ascending=False)
card.to_csv(OUT / "06_categorical_cardinality.csv", header=["n_unique"])
log(f"\n[6] {len(cat_cols)} cot dang chu. Nhieu gia tri nhat:\n{card.head(8).to_string()}")

# ---------------------------------------------------------------- 7. Tuong quan
num = df.select_dtypes("number").drop(columns=[LABEL, "TransactionID", "hour"], errors="ignore")
corr_y = num.corrwith(df[LABEL]).dropna()
top = corr_y.reindex(corr_y.abs().sort_values(ascending=False).index).head(30)
top.to_csv(OUT / "07_top_corr_with_label.csv", header=["corr"])
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(top.index[::-1], top.values[::-1], color="#4c78a8")
ax.set_title("30 cot so tuong quan manh nhat voi isFraud")
save(fig, "07_top_corr_with_label.png")
log(f"\n[7] Top 5 tuong quan voi nhan:\n{top.head(5).round(3).to_string()}")

sample = num.sample(min(50_000, len(num)), random_state=42)
c = sample.corr().abs().to_numpy()
iu = np.triu_indices_from(c, k=1)
n_pairs = int((c[iu] > 0.95).sum())
log(f"    So cap cot co |tuong quan| > 0.95: {n_pairs} -> nhieu cot du thua (dac biet nhom V1..V339)")

# ---------------------------------------------------------------- 8. Trung lap
dup = df.drop(columns=["TransactionID", "hour"], errors="ignore").duplicated().sum()
log(f"\n[8] So dong trung lap (bo TransactionID): {dup:,}")

(OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
log(f"\nXong. Hinh va bang nam trong {OUT.resolve()}")
