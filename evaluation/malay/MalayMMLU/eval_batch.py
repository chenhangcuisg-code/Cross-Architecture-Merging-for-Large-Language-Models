import os
import pandas as pd
import argparse


# ============================================================
# 1. 从路径解析实验超参数
#    - 有 alpha/lr → 正常解析
#    - 无 alpha/lr → baseline，alpha/lr=None（不 skip）
# ============================================================
def parse_from_path(pred_file):
    abs_path = os.path.abspath(pred_file)
    parts = abs_path.split(os.sep)

    filename = os.path.basename(pred_file).replace(".csv", "")
    name_parts = filename.split("_")
    if len(name_parts) < 3:
        return None  # 文件名不合法，才 skip

    eval_variant = name_parts[-3]
    by_letter = name_parts[-2]
    shot = name_parts[-1]

    alpha, lr = None, None
    alpha_dir_index = None

    for i, p in enumerate(parts):
        if "alpha" in p and "lr" in p:
            alpha_dir_index = i
            for seg in p.split("_"):
                if seg.startswith("alpha"):
                    alpha = seg.replace("alpha", "")
                elif seg.startswith("lr"):
                    lr = seg.replace("lr", "")

    # ---------- 解析 model / train_variant ----------
    if alpha_dir_index is not None:
        # 正常实验
        try:
            train_variant = parts[alpha_dir_index - 1]
            model = parts[alpha_dir_index - 2]
        except IndexError:
            return None
    else:
        # baseline / no alpha lr
        # e.g. .../llama3.2-1bma-instruct/baseline/modelA/xxx.csv
        try:
            model = parts[-4]
            train_variant = parts[-3]
        except IndexError:
            model = "unknown"
            train_variant = "baseline"

    return {
        "dataset": "MalayMMLU",
        "model": model,
        "train_variant": train_variant,
        "alpha": alpha,      # None → NaN
        "lr": lr,            # None → NaN
        "eval_variant": eval_variant,
        "by_letter": by_letter,
        "shot": shot,
        "filepath": abs_path,
    }


# ============================================================
# 2. 计算单个 category 的 accuracy
# ============================================================
def calculate_accuracy(mmlu, pred_file, keep_idxs):
    result = pd.read_csv(pred_file)
    by_letter = os.path.basename(pred_file).split("_")[-2]

    correct = 0
    for i in keep_idxs:
        if by_letter == "True":
            if result.iloc[i].preds == mmlu.iloc[i].key:
                correct += 1
        else:
            if result.iloc[i].preds == result.iloc[i].golds:
                correct += 1

    return correct / len(keep_idxs) * 100


# ============================================================
# 3. 收集 CSV
# ============================================================
def collect_files_from_directory(root_dir, ext=".csv"):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(ext):
                paths.append(os.path.join(root, f))
    return paths


# ============================================================
# 4. 按 category macro average 排序
# ============================================================
def sort_experiments_by_macro_average(df):
    acc_cols = [c for c in df.columns if c.startswith("accuracy_")]
    if not acc_cols:
        raise ValueError("No accuracy_* columns found")

    df = df.copy()
    df["_macro_avg"] = df[acc_cols].mean(axis=1)
    df = df.sort_values(by="_macro_avg", ascending=False)
    df = df.drop(columns=["_macro_avg"])
    return df


# ============================================================
# 5. 主逻辑
# ============================================================
def main(pred_files, shot, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mmlu = pd.read_json(f"data/MalayMMLU_{shot}shot.json")
    categories = sorted(mmlu.category.unique())

    rows = []

    for pred_file in pred_files:
        print(f"Processing: {pred_file}")
        meta = parse_from_path(pred_file)
        if meta is None:
            print(f"⚠️  Skip (invalid filename): {pred_file}")
            continue

        row = dict(meta)
        row["metric"] = "single"

        for cat in categories:
            keep_idxs = mmlu[mmlu.category == cat].index.tolist()
            acc = calculate_accuracy(mmlu, pred_file, keep_idxs)
            row[f"accuracy_{cat}"] = acc

        rows.append(row)

    df = pd.DataFrame(rows)

    # ✅ 防御：避免空 DF 直接炸
    if df.empty:
        print("❌ No valid experiments found. Nothing to save.")
        return

    # 列顺序
    front_cols = [
        "dataset", "model", "train_variant", "alpha", "lr",
        "eval_variant", "by_letter", "shot", "metric"
    ]
    acc_cols = sorted([c for c in df.columns if c.startswith("accuracy_")])

    df = df[front_cols + acc_cols + ["filepath"]]

    # ⭐ 从好到坏排序
    df = sort_experiments_by_macro_average(df)

    out_csv = os.path.join(output_dir, "accuracy_results_sorted.csv")
    df.to_csv(out_csv, index=False)

    print("\n✅ Final sorted CSV saved to:")
    print(out_csv)
    print("\nTop-5 experiments:")
    print(df.head(5))


# ============================================================
# 6. CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files", nargs="+", type=str)
    parser.add_argument("--pred_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shot", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    if args.all:
        assert args.pred_dir is not None
        files = collect_files_from_directory(args.pred_dir)
    else:
        files = args.pred_files

    main(files, args.shot, args.output_dir)
