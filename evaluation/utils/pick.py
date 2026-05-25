#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import pandas as pd
import numpy as np


def parse_lr_alpha(s: str):
    """
    从字符串里解析 alpha / lr
    例如: "alpha0.03_lr1e-5/xxx", "alpha0.01_lr1e-6"
    返回: (alpha: float or None, lr: str or None)
    """
    s = str(s)
    alpha, lr = None, None
    m = re.search(r"alpha([0-9.]+)_lr([0-9eE\-]+)", s)
    if m:
        try:
            alpha = float(m.group(1))
        except ValueError:
            alpha = None
        lr = m.group(2)
    return alpha, lr


def normalize_variant(existing_variant, source: str):
    """
    统一 variant 名称到 {hot, nohot, untrained, other}
    - 如果 csv 中有 variant 列，则优先用它并做归一化:
        hot -> hot
        nohot -> nohot
        ablation / untrained / baseline -> untrained
    - 如果没有，就从路径字符串里猜:
        包含 untrained / ablation -> untrained
        包含 nohot -> nohot
        有 hot 相关标记 -> hot
        否则 -> other
    """
    s = str(source).lower()

    # 有显式 variant 列
    if existing_variant is not None and not (
        isinstance(existing_variant, float) and np.isnan(existing_variant)
    ):
        name = str(existing_variant).strip().lower()
        mapping = {
            "hot": "hot",
            "nohot": "nohot",
            "untrained": "untrained",
            "ablation": "untrained",  # 新格式的 ablation 当 untrained
            "baseline": "untrained",
        }
        if name in mapping:
            return mapping[name]
        # 不认识的就继续从路径里猜

    # 从路径字符串里猜
    if "untrained" in s or "ablation" in s:
        return "untrained"
    if "nohot" in s:
        return "nohot"
    if "/hot/" in s or s.endswith("/hot") or re.search(r"[_/]hot[_/]", s):
        return "hot"

    return "other"


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一解析:
    - lr / alpha: 从 dir_path / folder_name / relative_path 里提取
    - variant: 归一化到 {hot, nohot, untrained, other}
    支持两种格式:
      1) 旧: 有 dir_path 列
      2) 新: 有 folder_name / variant / relative_path
    """
    df = df.copy()

    # 选择一个字段作为超参字符串来源
    if "dir_path" in df.columns:
        src_col = "dir_path"
    elif "folder_name" in df.columns:
        src_col = "folder_name"
    elif "relative_path" in df.columns:
        # 保底再用 relative_path
        src_col = "relative_path"
    else:
        raise ValueError("找不到 dir_path / folder_name / relative_path，无法解析 lr/alpha")

    has_variant_col = "variant" in df.columns

    lrs = []
    alphas = []
    variants = []

    for idx, row in df.iterrows():
        src = row[src_col]
        alpha, lr = parse_lr_alpha(src)
        lrs.append(lr)
        alphas.append(alpha)

        existing_variant = row["variant"] if has_variant_col else None
        v = normalize_variant(existing_variant, src)
        variants.append(v)

    df["lr"] = lrs
    df["alpha"] = alphas
    df["variant"] = variants

    return df


def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一打一个 score 列:
    优先级:
    1. mean_score (旧格式)
    2. token_f1    (新格式)
    3. exact_match
    4. 其它类似 xquad_th__f1,none 这类 metric 列的平均
    """
    df = df.copy()

    # 1) mean_score
    if "mean_score" in df.columns:
        score = pd.to_numeric(df["mean_score"], errors="coerce")
        if not score.isna().all():
            df["score"] = score
            return df

    # 2) token_f1
    if "token_f1" in df.columns:
        score = pd.to_numeric(df["token_f1"], errors="coerce")
        if not score.isna().all():
            df["score"] = score
            return df

    # 3) exact_match
    if "exact_match" in df.columns:
        score = pd.to_numeric(df["exact_match"], errors="coerce")
        if not score.isna().all():
            df["score"] = score
            return df

    # 4) metric 列平均 (旧格式那堆 ...__acc,none / ...__f1,none)
    metric_cols = [c for c in df.columns if "__" in c]
    if metric_cols:
        metrics = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        df["score"] = metrics.mean(axis=1)
        if not df["score"].isna().all():
            return df

    raise ValueError("找不到合适的指标列（mean_score / token_f1 / exact_match / metric 列）来计算 score")


def main(input_path: str, output_path: str):
    # 自动猜分隔符（逗号 / 制表符都可以）
    df = pd.read_csv(input_path, sep=None, engine="python")

    # 解析 lr / alpha / variant
    df = add_parsed_columns(df)

    # 计算 score
    df = compute_score(df)
    df = df[~df["score"].isna()].copy()

    # 只看 hot / nohot / untrained
    df = df[df["variant"].isin(["hot", "nohot", "untrained"])].copy()

    # ------- 第一步：nohot 的 lr -> 同 lr 里最烂的一条（baseline） -------
    nohot_df = df[(df["variant"] == "nohot") & df["lr"].notna()].copy()
    if nohot_df.empty:
        raise ValueError("没有任何 nohot 行（或没有解析出 lr），没法生成 lr list")

    # 按 score 升序（从差到好），每个 lr 只留最烂的一行
    nohot_worst_per_lr = (
        nohot_df
        .sort_values("score", ascending=True)
        .groupby("lr", as_index=False)
        .first()
    )
    # baseline 分数
    nohot_worst_per_lr["nohot_baseline_score"] = nohot_worst_per_lr["score"]

    lr_list = sorted(nohot_worst_per_lr["lr"].unique())

    # ------- 第二步：在这些 lr 里选“同 lr 下哪个 hot 最好（相对于 baseline）” -------
    hot_df = df[(df["variant"] == "hot") & df["lr"].notna()].copy()
    hot_df = hot_df[hot_df["lr"].isin(lr_list)].copy()

    # 合并 baseline
    hot_df = hot_df.merge(
        nohot_worst_per_lr[["lr", "nohot_baseline_score"]],
        on="lr",
        how="inner",
    )

    if hot_df.empty:
        raise ValueError("在这些 lr 下没有任何 hot 结果，没法做 hot vs nohot 对比")

    hot_df["diff_hot_vs_nohot"] = hot_df["score"] - hot_df["nohot_baseline_score"]

    # 每个 lr 内：
    #   先按 diff_hot_vs_nohot 降序，再按 hot 的绝对 score 降序，取第一条
    hot_best_per_lr = (
        hot_df
        .sort_values(["diff_hot_vs_nohot", "score"], ascending=[False, False])
        .groupby("lr", as_index=False)
        .first()
    )

    # ------- 第三步：nohot baseline + 最优 untrained -------
    # nohot baseline 就是上面 nohot_worst_per_lr
    nohot_selected = nohot_worst_per_lr.copy()

    # untrained：同 lr 下 score 最好的
    untrained_df = df[(df["variant"] == "untrained") & df["lr"].notna()].copy()
    untrained_df = untrained_df[untrained_df["lr"].isin(lr_list)].copy()

    if not untrained_df.empty:
        untrained_best_per_lr = (
            untrained_df
            .sort_values("score", ascending=False)
            .groupby("lr", as_index=False)
            .first()
        )
    else:
        # 没有 untrained 也没关系，给一个空 df
        untrained_best_per_lr = df.iloc[0:0].copy()

    # ------- 第四步：按“hot 相对 baseline 提升量”排序，输出 hot/nohot/untrained -------
    lr_rank_df = hot_best_per_lr[["lr", "diff_hot_vs_nohot", "score"]].copy()
    lr_rank_df = lr_rank_df.rename(columns={"score": "hot_score"})
    lr_rank_df = lr_rank_df.sort_values(
        ["diff_hot_vs_nohot", "hot_score"],
        ascending=[False, False],
    )
    lr_rank_df["group_rank"] = range(1, len(lr_rank_df) + 1)

    # 拼三类
    selected = pd.concat(
        [hot_best_per_lr, nohot_selected, untrained_best_per_lr],
        ignore_index=True,
        sort=False,
    )

    # 只保留有 hot 的那些 lr（因为只有这些 lr 才有 group_rank）
    selected = selected.merge(
        lr_rank_df[["lr", "group_rank"]],
        on="lr",
        how="inner",
    )

    # 组内排序：hot -> nohot -> untrained
    variant_order_map = {"hot": 0, "nohot": 1, "untrained": 2}
    selected["variant_order"] = selected["variant"].map(variant_order_map).fillna(99)

    selected = selected.sort_values(
        ["group_rank", "variant_order"],
        ascending=[True, True],
    )

    # 去掉辅助排序列，保留 diff / nohot_baseline_score 方便看
    selected = selected.drop(columns=["variant_order", "group_rank"])

    selected.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据 hot / nohot / untrained(auto: ablation) 自动选最优 lr/alpha 并排序输出"
    )
    args = parser.parse_args()

    main('eval_results_indo.csv', 'eval_results_indo_pick.csv')
