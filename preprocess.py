"""
Tien xu ly (pre-processing) IEEE-CIS Fraud Detection.

QUY TAC QUAN TRONG: moi tham so (median, cot bi bo, tan suat, mean/std...) chi duoc
HOC TREN TAP TRAIN, roi ap dung sang val va test. Neu hoc tren toan bo du lieu se bi
ro ri thong tin (data leakage).

Dau vao : data/processed/{train,val,test}.parquet
Dau ra  : data/clean/{train,val,test}.parquet          (da dien thieu, KHONG scale -> cho cay: RF, XGBoost)
          data/clean/{train,val,test}_scaled.parquet   (log1p cot lech + chuan hoa -> cho LR, MLP, Autoencoder)
          data/clean/preprocess_artifacts.pkl          (tham so da hoc, de tai su dung)
          data/clean/dropped_columns.csv               (cot bi bo + ly do, de dua vao bao cao)

Luu y: BUOC XU LY MAT CAN BANG (SMOTE, class weight...) KHONG lam o day;
       chi ap dung len tap train, do TV3 lam sau.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

SRC = Path("data/processed")
DST = Path("data/clean")
DST.mkdir(parents=True, exist_ok=True)
LABEL = "isFraud"

MISSING_DROP = 0.90      # bo cot thieu > 90% (tinh tren train)
ONEHOT_MAX = 10          # cot chu co <= 10 gia tri -> one-hot; nhieu hon -> ma hoa tan suat
CORR_DROP = 0.95         # bo cot co |tuong quan| > 0.95 voi mot cot dung truoc
SKEW_LOG = 3.0           # cot khong am co do lech > 3 -> log1p (chi ban scaled)
CLIP_Z = 10.0            # cat gia tri scaled ve [-10, 10] cho on dinh khi huan luyen


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tao feature moi tu tung dong (khong hoc gi tu du lieu -> khong ro ri)."""
    df = df.copy()
    df["n_missing"] = df.isna().sum(axis=1).astype("int16")      # so o thieu trong dong
    df["has_identity"] = df["id_01"].notna().astype("int8")       # co thong tin thiet bi
    dt = df["TransactionDT"]
    df["hour"] = ((dt // 3600) % 24).astype("int8")
    df["dow"] = ((dt // 86400) % 7).astype("int8")
    df["log_amt"] = np.log1p(df["TransactionAmt"]).astype("float32")
    df["amt_frac"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype("float32")
    p, r = df["P_emaildomain"], df["R_emaildomain"]
    df["email_match"] = ((p == r) & p.notna()).astype("int8")
    return df.drop(columns=["TransactionID", "TransactionDT"])


class Preprocessor:
    def fit(self, train: pd.DataFrame):
        X = add_features(train).drop(columns=[LABEL])
        self.dropped = []  # (cot, ly do)

        # 1) bo cot thieu qua nhieu
        miss = X.isna().mean()
        for c in miss[miss > MISSING_DROP].index:
            self.dropped.append((c, f"thieu > {MISSING_DROP:.0%}"))
        X = X.drop(columns=[c for c, _ in self.dropped])

        # 2) bo cot hang so
        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        for c in const:
            self.dropped.append((c, "hang so"))
        X = X.drop(columns=const)

        # 3) phan loai cot: chu vs so (pandas moi dung kieu 'str', nen dung exclude='number')
        cat_cols = list(X.select_dtypes(exclude="number").columns)
        num_cols = list(X.select_dtypes(include="number").columns)

        # 4) bo cot so trung lap (tuong quan cao) - tinh tren mau 50k dong cua train
        s = X[num_cols].sample(min(50_000, len(X)), random_state=42)
        corr = s.corr().abs().to_numpy()
        corr_drop = []
        for j in range(len(num_cols)):
            if num_cols[j] in corr_drop:
                continue
            for k in range(j + 1, len(num_cols)):
                if corr[j, k] > CORR_DROP and num_cols[k] not in corr_drop:
                    corr_drop.append(num_cols[k])
        for c in corr_drop:
            self.dropped.append((c, f"tuong quan > {CORR_DROP} voi cot khac"))
        self.num_cols = [c for c in num_cols if c not in corr_drop]

        # 5) cot chu: one-hot (it gia tri) hoac ma hoa tan suat (nhieu gia tri)
        self.onehot = {}
        self.freq = {}
        for c in cat_cols:
            v = X[c].fillna("__nan__").astype(str)
            if v.nunique() <= ONEHOT_MAX:
                self.onehot[c] = sorted(v.unique())
            else:
                self.freq[c] = v.value_counts(normalize=True).to_dict()

        # 6) median de dien thieu (tu train)
        self.medians = X[self.num_cols].median()

        # 7) cot lech de log1p + mean/std de chuan hoa (tu train, tren du lieu da dien thieu)
        filled = X[self.num_cols].fillna(self.medians)
        skew = filled.skew()
        self.log_cols = [c for c in self.num_cols if filled[c].min() >= 0 and skew[c] > SKEW_LOG]
        scaled_part = self._build_scaled_inputs(filled, self._freq_frame(X))
        self.mean = scaled_part.mean()
        self.std = scaled_part.std(ddof=0).replace(0, 1.0)
        return self

    # ---- ham phu
    def _freq_frame(self, X):
        out = {}
        for c, m in self.freq.items():
            out[f"{c}__freq"] = X[c].fillna("__nan__").astype(str).map(m).fillna(0.0).astype("float32")
        return pd.DataFrame(out, index=X.index)

    def _onehot_frame(self, X):
        out = {}
        for c, cats in self.onehot.items():
            v = X[c].fillna("__nan__").astype(str)
            for cat in cats:
                out[f"{c}__{cat}"] = (v == cat).astype("int8")
        return pd.DataFrame(out, index=X.index)

    def _build_scaled_inputs(self, num_df, freq_df):
        num_df = num_df.copy()
        for c in self.log_cols:
            num_df[c] = np.log1p(num_df[c].clip(lower=0))
        return pd.concat([num_df, freq_df], axis=1)

    # ---- ap dung
    def transform(self, df: pd.DataFrame, scale: bool):
        X = add_features(df)
        y = X.pop(LABEL).astype("int8")
        num = X[self.num_cols].fillna(self.medians)
        freq = self._freq_frame(X)
        onehot = self._onehot_frame(X)
        if scale:
            cont = self._build_scaled_inputs(num, freq)
            cont = ((cont - self.mean) / self.std).clip(-CLIP_Z, CLIP_Z)
        else:
            cont = pd.concat([num, freq], axis=1)
        out = pd.concat([cont, onehot], axis=1).astype("float32")
        out[LABEL] = y
        return out


if __name__ == "__main__":
    raw = {s: pd.read_parquet(SRC / f"{s}.parquet") for s in ["train", "val", "test"]}

    pre = Preprocessor().fit(raw["train"])        # CHI hoc tren train
    pd.DataFrame(pre.dropped, columns=["column", "reason"]).to_csv(DST / "dropped_columns.csv", index=False)
    with open(DST / "preprocess_artifacts.pkl", "wb") as f:
        pickle.dump(pre, f)

    print("=== Cot bi bo ===")
    reasons = pd.DataFrame(pre.dropped, columns=["column", "reason"])["reason"].value_counts()
    print(reasons.to_string())
    print(f"\nCot so giu lai: {len(pre.num_cols)} | one-hot: {len(pre.onehot)} cot chu | ma hoa tan suat: {len(pre.freq)} cot chu")
    print(f"Cot duoc log1p (ban scaled): {len(pre.log_cols)}")

    for split, df in raw.items():
        for scale, suffix in [(False, ""), (True, "_scaled")]:
            out = pre.transform(df, scale)
            out.to_parquet(DST / f"{split}{suffix}.parquet", index=False)
            if suffix == "":
                n_feat = out.shape[1] - 1
                print(f"{split:5s}: {out.shape[0]:>7,} mau, {n_feat} feature, fraud={out[LABEL].mean():.3%}, "
                      f"NaN con lai={int(out.isna().sum().sum())}")

    print("\nXong. File nam trong", DST.resolve())
