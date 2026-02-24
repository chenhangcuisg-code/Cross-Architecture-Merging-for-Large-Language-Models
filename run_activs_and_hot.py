# -*- coding: utf-8 -*-
"""
规范化版 run_activs_and_hot.py

- 不再有 pipeline 选项：
    data = "malay" / "indonesian" / "eng" / "medical" -> 文本流程
    data = "gsm8k"                                    -> GSM8K 流程

- 不再有 model-a-label / model-b-label 选项。

- 数据集参数：
    --data / --data-subset:  malay | gsm8k | indonesian | eng | medical
    --data-split:            train / test / ...

其它参数：
    --model-a-path / --model-b-path
    --batch-size-a / --batch-size-b
    --device-a / --device-b
    --top-neuron-dir-a / --top-neuron-dir-b  (只在 malay/indonesian/eng/medical 文本流程用)
    --hot-chunk-cols / --hot-dtype / --out-dir
    --model-a-type / --model-b-type (只在 data=gsm8k 时用，控制 A/B 是 tinyllava / qwenvl* / llama3 / qwen2)
"""

import argparse
import os
import sys
import traceback
from typing import Dict, List

import torch
import transformers

# 统一数据加载
from dataset_hot_texts import (
    build_fineweb_thai_chat_texts,
    load_texts,
)

# Llama3 / Qwen2 采集函数（文本流程 gen_qwen 等只用这两个，不依赖 TinyLLaVA/QwenVL）
from activs_llama3_modules import (
    get_llama3_block_activations_by_module as get_llama3_block_activations_plain,
    get_llama3_block_activations_by_module_neuron,
)
from activs_qwen2_modules import get_qwen2_block_activations_by_module_neuron

# Qwen2-VL / TinyLLaVA 仅在 data=gsm8k 或 model_type=tinyllava/qwenvl 时按需导入，避免 apply_chunking_to_forward 等兼容问题

# HOT 计算（共用，submission 使用 core 包）
from core.hot_transport_chunk import (
    compute_Q_and_layer_costs,
    compute_P_stable as compute_P,
    reconstruct_X,
)


# -------------------- 通用小工具 --------------------
def print_coverage(per_layer_dict, model_name: str) -> None:
    print(f"\n[Coverage report for {model_name}]")
    for kind, layers in per_layer_dict.items():
        counts = []
        for v in layers:
            if isinstance(v, torch.Tensor):
                counts.append(int(v.shape[0]))  # T（样本/拼接数）
            elif isinstance(v, list):
                counts.append(len(v))
            else:
                counts.append(0)
        total = sum(counts)
        nonzero = sum(1 for c in counts if c > 0)
        print(
            f"  {kind:<12s}: total={total:<6d} "
            f"layers_with_data={nonzero}/{len(counts)}  per_layer={counts}"
        )


def _has_key(d, k):
    return (k in d) and isinstance(d[k], list) and len(d[k]) > 0


def get_module_per_layer(per_layer_dict, key: str):
    """
    返回该模块的 per-layer 列表；兼容 fwd_up/fwd_down 别名；支持 *_pre 直接取。
    """
    if key in per_layer_dict:
        return per_layer_dict[key]
    if key == "up" and _has_key(per_layer_dict, "fwd_up"):
        return per_layer_dict["fwd_up"]
    if key == "down" and _has_key(per_layer_dict, "fwd_down"):
        return per_layer_dict["fwd_down"]
    if key == "up_pre" and _has_key(per_layer_dict, "fwd_up_pre"):
        return per_layer_dict["fwd_up_pre"]
    if key == "down_pre" and _has_key(per_layer_dict, "fwd_down_pre"):
        return per_layer_dict["fwd_down_pre"]
    raise KeyError(f"模块 {key} 在 per_layer 中不存在。")


def filter_nonempty_layer_pairs(X_list, Y_list):
    """去掉任一侧为空/numel=0 的层，避免 HOT 报错；返回过滤后的列表与保留的层索引。"""
    keep_idx = []
    for li, (XA, YB) in enumerate(zip(X_list, Y_list)):
        ok = (
            isinstance(XA, torch.Tensor)
            and isinstance(YB, torch.Tensor)
            and XA.numel() > 0
            and YB.numel() > 0
        )
        if ok:
            keep_idx.append(li)
    Xf = [X_list[i] for i in keep_idx]
    Yf = [Y_list[i] for i in keep_idx]
    return Xf, Yf, keep_idx


def normalize_and_get_sigma(x: torch.Tensor):
    """
    按特征维标准化：每层 (T,H) → (T,H)，同时返回每个特征的 std:
      输入:  x (T,H)
      输出: x_norm (T,H), sigma (H,)
    """
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return x, None
    x = x.to(torch.float32)
    mu = x.mean(dim=0, keepdim=True)
    sigma = x.std(dim=0, keepdim=True) + 1e-6  # (1,H)
    x_norm = (x - mu) / sigma  # (T,H)
    return x_norm, sigma.squeeze(0)  # sigma -> (H,)


def normalize_per_feature(x: torch.Tensor) -> torch.Tensor:
    """代码A用的简单标准化：仅返回标准化后的张量。"""
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return x
    x = x.to(torch.float32)
    mu = x.mean(dim=0, keepdim=True)
    sigma = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mu) / sigma


# -------------------- argparse --------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HOT transport between two models with activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型路径与设备
    parser.add_argument(
        "--model-a-path",
        type=str,
        default=None,
        help="模型 A 的本地路径或 HF 名称。",
    )
    parser.add_argument(
        "--model-b-path",
        type=str,
        default=None,
        help="模型 B 的本地路径或 HF 名称。",
    )

    parser.add_argument(
        "--batch-size-a",
        type=int,
        default=2,
        help="模型 A 前向 batch size。",
    )
    parser.add_argument(
        "--batch-size-b",
        type=int,
        default=2,
        help="模型 B 前向 batch size。",
    )
    parser.add_argument(
        "--device-a",
        type=str,
        default="cuda:0",
        help="模型 A 使用的设备（如 cuda:0 / cpu）。",
    )
    parser.add_argument(
        "--device-b",
        type=str,
        default="cuda:0",
        help="模型 B 使用的设备（如 cuda:0 / cpu）。",
    )

    # top neuron 目录（只在 malay/indonesian/eng/medical 文本流程使用）
    parser.add_argument(
        "--top-neuron-dir-a",
        type=str,
        default=None,
        help="模型 A 使用的 top neuron 目录；若不需要可设为 '' 或 None。",
    )
    parser.add_argument(
        "--top-neuron-dir-b",
        type=str,
        default=None,
        help="模型 B 的 top neuron 目录；通常为 None。",
    )

    # 数据集配置：data 决定走哪条流程
    parser.add_argument(
        "--data",
        "--data-subset",
        dest="data",
        choices=["malay", "gsm8k", "indonesian", "eng", "medical", "alpaca", "fineweb_thai", "finance", "cantonese"],
        default="malay",
        help=(
            "数据集选择："
            "malay / indonesian / eng / cantonese / medical / alpaca / fineweb_thai / finance 使用通用文本流程；"
            "gsm8k 使用 GSM8K 特殊流程。"
        ),
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="train",
        help="数据 split，例如 train / test。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="最多使用多少条文本样本。",
    )

    # HOT 参数 & 输出目录（chunk_cols / dtype 只记录在 notes 中）
    parser.add_argument(
        "--hot-chunk-cols",
        type=int,
        default=1024,
        help="HOT 计算的列分块大小（仅用于记录说明；真正分块逻辑在 hot_transport_chunk 内）。",
    )
    parser.add_argument(
        "--hot-dtype",
        type=str,
        default="float32",
        help="HOT 计算使用的数据类型（仅用于记录说明）。",
    )
    parser.add_argument(
        "--hot-save-dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "uint8"),
        help="保存到 .pt 时 Q_list 的数据类型。float16 约 1/2 体积，uint8 约 1/4 体积；融合时自动转回 float32。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出 .pt 文件保存目录。",
    )

    # 仅对 data=gsm8k 流程有意义的额外配置（保持原代码A的行为）
    parser.add_argument(
        "--model-a-type",
        type=str,
        default=None,
        help=(
            "仅 data=gsm8k 时使用：A 侧模型类型。"
            "推荐取值：tinyllava / qwenvl2 / qwenvl2.5 / llama3 等；"
            "内部通过包含关系判断（如包含 'tinyllava' 就认为是 TinyLLaVA，"
            "包含 'qwenvl' 就认为是 Qwen2-VL 系列）。"
            "文本数据 (malay/indonesian/eng/medical) 流程会忽略该参数。"
        ),
    )
    parser.add_argument(
        "--model-b-type",
        type=str,
        default=None,
        help=(
            "仅 data=gsm8k 时使用：B 侧模型类型。"
            "推荐取值：llama3 / qwen2 / qwenvl2 / qwenvl2.5 等；"
            "内部通过包含关系判断（如包含 'qwenvl' 就认为是 Qwen2-VL 系列）。"
            "文本数据 (malay/indonesian/eng/medical) 流程会忽略该参数。"
        ),
    )

    return parser


# -------------------- 文本数据: malay / indonesian / eng / medical (原代码B) --------------------
def run_pipeline_llama3_text(args: argparse.Namespace) -> None:
    """
    文本数据流程（malay / indonesian / eng / cantonese / medical）：
    现在支持 A/B 侧任意模型类型：
        - Llama3（默认）
        - TinyLLaVA      （自动识别：model-path 或 model-type 含 tinyllava）
        - Qwen2-VL       （自动识别：路径含 qwenvl）
        - Qwen2          （自动识别：路径含 qwen2，但不含 vl）
    """

    # ---------- 0) 解析模型类型 ----------
    def detect_model_type(path: str):
        p = (path or "").lower()
        if "tinyllava" in p:
            return "tinyllava"
        if "qwenvl" in p:
            return "qwenvl"
        if "qwen3" in p:
            return "qwen2"  # Qwen3 与 Qwen2 同用 qwen2 采集逻辑（trust_remote_code 加载）
        if "qwen2" in p:
            return "qwen2"
        return "llama3"

    # 使用相对路径或环境变量，无实名/绝对路径默认值
    if args.model_a_path is None:
        args.model_a_path = os.environ.get("MODEL_A_PATH", "models/source_1b")
    if args.model_b_path is None:
        args.model_b_path = os.environ.get("MODEL_B_PATH", "models/target_8b")

    typA = detect_model_type(args.model_a_path)
    typB = detect_model_type(args.model_b_path)

    if args.out_dir is None:
        args.out_dir = os.environ.get("HOT_OUT_DIR", "transport_results/llama3_text")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) 加载文本（统一入口） ----------
    raw_texts = load_texts(
        data=args.data,
        split=args.data_split,
        max_samples=args.max_samples,
    )

    fineweb_mode = args.data in {"fineweb_thai", "thai_fineweb"}
    template_cache: Dict[str, List[str]] = {}

    def _model_type_to_template(model_type: str) -> str:
        m = (model_type or "").lower()
        if "qwenvl" in m or "qwen2" in m or "qwen3" in m:
            return "qwen2"
        return "llama3"

    def _prompts_for_model_type(model_type: str) -> list[str]:
        if not fineweb_mode:
            return raw_texts
        template_name = _model_type_to_template(model_type)
        if template_name not in template_cache:
            print(f"[Info] Formatting fineweb_thai texts for template={template_name}")
            template_cache[template_name] = build_fineweb_thai_chat_texts(
                raw_texts,
                template=template_name,
            )
        return template_cache[template_name]

    # ---------- 2) 定义统一的激活采集 wrapper ----------
    def get_activs(model_path, model_type, batch_size, device, top_dir):
        prompts = _prompts_for_model_type(model_type)
        if model_type == "llama3":
            return get_llama3_block_activations_by_module_neuron(
                prompts=prompts,
                model_name=model_path,
                batch_size=batch_size,
                device=device,
                top_neuron_dir=top_dir,
            )
        elif model_type == "tinyllava":
            from activs_tinyllava_modules import get_tinyllava_block_activations_by_module
            return get_tinyllava_block_activations_by_module(
                model_path=model_path,
                texts=prompts,
                conv_mode="qwen2_base",
                image_paths=None,
                insert_image_token=None,
                batch_size=batch_size,
                device=device,
            )
        elif model_type == "qwenvl":
            from activs_qwenvl2_modules import get_qwenvl2_block_activations_by_module
            return get_qwenvl2_block_activations_by_module(
                model_name=model_path,
                texts=prompts,
                image_paths=None,
                batch_size=batch_size,
                device=device,
                exclude_image_token=True,
            )
        elif model_type == "qwen2":
            return get_qwen2_block_activations_by_module_neuron(
                prompts=prompts,
                model_name=model_path,
                batch_size=batch_size,
                device=device,
                top_neuron_dir=top_dir
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ---------- 3) 抓取 A/B 侧激活 ----------
    print(f"[Info] A-side model={args.model_a_path} ({typA})")
    xa_per_layer, xa_per_module = get_activs(
        args.model_a_path,
        typA,
        args.batch_size_a,
        args.device_a,
        args.top_neuron_dir_a,
    )

    # 清理GPU内存，避免A侧模型占用内存影响B侧模型加载
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[Info] Cleared GPU cache after A-side model processing")

    print(f"[Info] B-side model={args.model_b_path} ({typB})")
    xb_per_layer, xb_per_module = get_activs(
        args.model_b_path,
        typB,
        args.batch_size_b,
        args.device_b,
        args.top_neuron_dir_b,
    )

    print_coverage(xa_per_layer, "A")
    print_coverage(xb_per_layer, "B")

    # -------------------- 4) 模块配置 --------------------
    BASE_MODULES = ["K", "Q", "V", "O"]
    MODULES_WANTED = []
    for m in BASE_MODULES:
        MODULES_WANTED += [m, m + "_pre"]  # post 与 pre 都计算

    BIG_FFN = {
        "fwd_up",
        "fwd_down",
        "fwd_up_pre",
        "fwd_down_pre",
        "up",
        "down",
        "up_pre",
        "down_pre",
    }

    summary = {}

    # -------------------- 5) HOT 计算与保存 --------------------
    for mod in MODULES_WANTED:
        # 设备策略：FFN/大矩阵改用 CPU（分块下等价），注意力依然用 GPU
        device_local = "cuda:0"
        if (mod in BIG_FFN) or ("fwd" in mod):
            device_local = "cpu"

        # 取两侧每层激活
        try:
            X_list_raw = get_module_per_layer(xa_per_layer, mod)
            Y_list_raw = get_module_per_layer(xb_per_layer, mod)
        except KeyError:
            print(f"[Skip] {mod}: 缺少该键。")
            continue

        print(f"\n[Diag] module={mod}")
        for li, (XA, YB) in enumerate(zip(X_list_raw[:3], Y_list_raw[:3])):
            if isinstance(XA, torch.Tensor) and isinstance(YB, torch.Tensor):
                print(
                    f"  layer {li}: A T×H={tuple(XA.shape)}, "
                    f"B T×H={tuple(YB.shape)}"
                )
            else:
                print(f"  layer {li}: A/B 其中至少一侧为空")

        # 过滤空层
        X_list, Y_list, kept = filter_nonempty_layer_pairs(X_list_raw, Y_list_raw)
        if len(kept) == 0:
            print(f"[WARN] {mod}: 所有层均为空，跳过 HOT。")
            continue
        if len(kept) < len(X_list_raw):
            dropped = [i for i in range(len(X_list_raw)) if i not in kept]
            print(f"[INFO] {mod}: 过滤掉空层 {dropped}，实际计算层数={len(kept)}")

        # 标准化，同时记录每层的 std（后面用于将 Q 从 z-score 空间还原到 raw 空间）
        X_list_norm, sigA_list = [], []
        Y_list_norm, sigB_list = [], []
        for x in X_list:
            x_n, sig = normalize_and_get_sigma(x)
            X_list_norm.append(x_n)
            sigA_list.append(sig)  # (H_A,) or None
        for y in Y_list:
            y_n, sig = normalize_and_get_sigma(y)
            Y_list_norm.append(y_n)
            sigB_list.append(sig)  # (H_B,) or None

        X_list = X_list_norm
        Y_list = Y_list_norm

        # 在 HOT 计算前，检查某一层的 mean/std（防止越界，取最后一层也行）
        layer_id = min(10, len(X_list) - 1)
        XA = X_list[layer_id]
        YB = Y_list[layer_id]
        print(
            f"Layer {layer_id} A mean/std:",
            XA.mean().item(),
            XA.std().item(),
        )
        print(
            f"Layer {layer_id} B mean/std:",
            YB.mean().item(),
            YB.std().item(),
        )

        # 数值稳定的 HOT 计算（带自适应正则）
        with torch.no_grad():
            reg_inner = 0.1
            reg_outer = 0.1

            # 注意：compute_Q_and_layer_costs 内部已做列分块与 dtype 控制
            Q_list_mod, C_layer_mod, _ = compute_Q_and_layer_costs(
                X_list,
                Y_list,
                reg_inner=reg_inner,
                device=device_local,
            )

            # cost 统计
            c_min = C_layer_mod.min().item()
            c_max = C_layer_mod.max().item()
            c_mean = C_layer_mod.mean().item()
            print(
                f"[{mod}] cost stats: min={c_min:.2f}, "
                f"max={c_max:.2f}, mean={c_mean:.2f}"
            )

            # 若 cost 偏大，放大 reg_outer
            if c_max > 1e3:
                scale = c_max / 100.0
                reg_outer = reg_outer * scale
                print(
                    f"[{mod}] ⚠️ cost 太大({c_max:.1f})，调整 "
                    f"reg_outer × {scale:.1f} → {reg_outer:.2e}"
                )

            # 计算 P
            P_mod = compute_P(C_layer_mod, reg_outer=reg_outer)

            # P 诊断
            p_min = P_mod.min().item()
            p_max = P_mod.max().item()
            p_sum = P_mod.sum().item()
            print(
                f"[{mod}] P stats: min={p_min:.3e}, "
                f"max={p_max:.3e}, sum={p_sum:.3e}"
            )

            if p_sum < 1e-6:
                reg_outer *= 10
                print(
                    f"[{mod}] ⚠️ P 全为 0，放大 reg_outer 重算 → "
                    f"{reg_outer:.2e}"
                )
                P_mod = compute_P(C_layer_mod, reg_outer=reg_outer)
                print(
                    f"[{mod}] Recomputed P stats: "
                    f"min={P_mod.min():.3e}, "
                    f"max={P_mod.max():.3e}, "
                    f"sum={P_mod.sum():.3e}"
                )

        # ------ 将 Q 从 z-score 空间转换为 raw 空间 ------
        # Q_raw[i,j] = Q[i,j] * sigma_X_i[:,None] / sigma_Y_j[None,:]
        Q_list_mod_raw = []
        for i, row in enumerate(Q_list_mod):
            sigA = sigA_list[i]  # (H_A,) or None
            row_new = []
            for j, q in enumerate(row):
                if not isinstance(q, torch.Tensor) or q.numel() == 0:
                    row_new.append(q)
                    continue

                sigB = sigB_list[j]  # (H_B,) or None
                if sigA is None or sigB is None:
                    row_new.append(q)
                    continue

                q = q.to(torch.float32)
                sigA_dev = sigA.to(q.device, dtype=q.dtype)  # (H_A,)
                sigB_dev = sigB.to(q.device, dtype=q.dtype)  # (H_B,)

                # ---- 统计 Q_norm ----
                q_norm_min = q.min().item()
                q_norm_max = q.max().item()
                q_norm_mean = q.mean().item()
                q_norm_std = q.std().item()

                # ---- 放缩比例统计 ----
                scale = sigA_dev.view(-1, 1) / (sigB_dev.view(1, -1) + 1e-6)
                scale_min = scale.min().item()
                scale_max = scale.max().item()
                scale_mean = scale.mean().item()

                # ---- 计算 Q_raw ----
                # 注意：这里保持与原始版本一致（实际上应为 q * scale）
                q_raw = q 
                # ---- 统计 Q_raw ----
                q_raw_min = q_raw.min().item()
                q_raw_max = q_raw.max().item()
                q_raw_mean = q_raw.mean().item()
                q_raw_std = q_raw.std().item()

                # ---- 打印转换前后的对比 ----
                print(f"[Q convert] layerA={i}, layerB={j}")
                print(
                    f"    Q_norm: min={q_norm_min:.3e}, "
                    f"max={q_norm_max:.3e}, "
                    f"mean={q_norm_mean:.3e}, "
                    f"std={q_norm_std:.3e}"
                )
                print(
                    f"    scale:  min={scale_min:.3e}, "
                    f"max={scale_max:.3e}, "
                    f"mean={scale_mean:.3e}"
                )
                print(
                    f"    Q_raw : min={q_raw_min:.3e}, "
                    f"max={q_raw_max:.3e}, "
                    f"mean={q_raw_mean:.3e}, "
                    f"std={q_raw_std:.3e}"
                )
                print("-" * 60)

                row_new.append(q_raw)

            Q_list_mod_raw.append(row_new)

        # -------------------------------------------------
        # 保存（文件名带上 _pre 后缀）
        # 体积说明：P 只有 (L,M) 很小；占大头的是 Q_list：L×M 个矩阵，每个 (H_A,H_B)，即每对层一个 neuron-neuron transport
        suffix = "" if not mod.endswith("_pre") else "_pre"
        mod_base = mod[:-4] if suffix == "_pre" else mod
        save_path = os.path.join(args.out_dir, f"hot_{mod_base}{suffix}.pt")
        save_dtype_str = getattr(args, "hot_save_dtype", "float32")
        save_dtype = torch.float32
        use_uint8 = save_dtype_str == "uint8"
        if save_dtype_str == "float16":
            save_dtype = torch.float16

        def _quantize_q_list(q_list_raw):
            if use_uint8:
                q_out = []
                scales_out = []
                for row in q_list_raw:
                    q_row = []
                    scale_row = []
                    for q in row:
                        if q is None or not isinstance(q, torch.Tensor):
                            q_row.append(None)
                            scale_row.append(None)
                            continue
                        q = q.cpu().float()
                        scale = q.max().clamp_min(1e-8).item()
                        q_u8 = (q / scale * 255.0).clamp(0.0, 255.0).round().to(torch.uint8)
                        q_row.append(q_u8)
                        scale_row.append(scale)
                    q_out.append(q_row)
                    scales_out.append(scale_row)
                return q_out, scales_out
            else:
                out_list = [
                    [(q.cpu().to(save_dtype) if isinstance(q, torch.Tensor) else None) for q in row]
                    for row in q_list_raw
                ]
                return out_list, None

        Q_list_saved, Q_scales = _quantize_q_list(Q_list_mod_raw)
        out = {
            "module": mod,
            "P": P_mod.cpu().to(save_dtype),
            "Q_list": Q_list_saved,
            "C_layer": C_layer_mod.cpu().to(save_dtype),
            "L": len(X_list),
            "M": len(Y_list),
            "kept_layer_idx": kept,
            **({"Q_scales": Q_scales} if Q_scales is not None else {}),
            "sigma_A": [
                s.cpu() if isinstance(s, torch.Tensor) else None for s in sigA_list
            ],
            "sigma_B": [
                s.cpu() if isinstance(s, torch.Tensor) else None for s in sigB_list
            ],
            "notes": (
                f"A={args.model_a_path}, B={args.model_b_path}；"
                + ("post（线性层之后）" if suffix == "" else "pre（线性层之前）")
                + f"；chunk_cols={args.hot_chunk_cols} dtype={args.hot_dtype} "
                f"device={device_local}；"
                + "Q 已映射到 raw 激活空间（乘 sigma_A / 除 sigma_B）"
            ),
        }
        
        # 确保输出目录存在
        try:
            os.makedirs(args.out_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] {mod}: 无法创建输出目录 {args.out_dir}: {e}", file=sys.stderr)
            continue
        
        # 保存文件，带异常处理
        try:
            torch.save(out, save_path)
            # 验证文件是否成功保存
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"文件保存后不存在: {save_path}")
            file_size = os.path.getsize(save_path)
            if file_size == 0:
                raise ValueError(f"保存的文件大小为0: {save_path}")
        except Exception as e:
            print(f"[ERROR] {mod}: 保存失败 -> {save_path}", file=sys.stderr)
            print(f"  错误详情: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

        # 控制台简报
        rs = P_mod.sum(dim=1).detach().cpu().tolist()
        cs = P_mod.sum(dim=0).detach().cpu().tolist()
        print(f"[{mod}] saved -> {save_path}")
        print(
            f"  L={len(X_list)}, M={len(Y_list)}, "
            f"P row sums≈ {rs[:3]}{'...' if len(rs) > 3 else ''}"
        )
        print(
            f"                               col sums≈ "
            f"{cs[:3]}{'...' if len(cs) > 3 else ''}"
        )

        summary[mod] = {
            "L": len(X_list),
            "M": len(Y_list),
            "kept": kept,
        }

    print("Done. Summary:", summary)


# -------------------- GSM8K: data=gsm8k (原代码A + B 侧 qwenvl 支持) --------------------
def run_pipeline_gsm8k(args: argparse.Namespace) -> None:
    # 默认模型类型；若提供了路径则自动推断，避免误判为 TinyLLaVA
    def _detect_model_type(path: str):
        p = (path or "").lower()
        if "tinyllava" in p:
            return "tinyllava"
        if "qwenvl" in p:
            return "qwenvl"
        # 兼容带连字符/下划线的 Llama-3 变体名称
        if "llama3" in p or "llama-3" in p or "llama_3" in p:
            return "llama3"
        return None

    if args.model_a_type is None:
        guessed = _detect_model_type(args.model_a_path)
        args.model_a_type = guessed if guessed is not None else "tinyllava"  # 原始代码A默认 TinyLLaVA
    if args.model_b_type is None:
        args.model_b_type = "llama3"  # 原始代码A默认 Llama3

    # 默认模型路径（可被命令行 override）
    if args.model_a_path is None:
        t_a = args.model_a_type.lower()
        if "tinyllava" in t_a:
            args.model_a_path = "Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP"
        elif "qwenvl" in t_a:
            args.model_a_path = os.environ.get("MODEL_A_PATH", "Qwen/Qwen2-VL-2B-Instruct")
        elif "llama3" in t_a:
            args.model_a_path = os.environ.get("MODEL_A_PATH", "meta-llama/Llama-3.2-1B-Instruct")
        else:
            raise ValueError(
                "data=gsm8k 时，model-a-type 仅支持包含 'tinyllava'、'llama3' 或 'qwenvl' 的字符串"
            )

    if args.model_b_path is None:
        t_b = args.model_b_type.lower()
        if "llama3" in t_b:
            args.model_b_path = os.environ.get("MODEL_B_PATH", "meta-llama/Llama-3.1-8B-Instruct")
        elif "qwenvl" in t_b:
            args.model_b_path = os.environ.get("MODEL_B_PATH", "Qwen/Qwen2-VL-2B-Instruct")
        elif t_b.startswith("qwen2"):
            args.model_b_path = "Qwen/Qwen2-7B-Instruct"
        else:
            raise ValueError(
                "data=gsm8k 时，model-b-type 仅支持包含 'llama3'、'qwen2' 或 'qwenvl' 的字符串"
            )

    if args.out_dir is None:
        args.out_dir = os.environ.get("HOT_OUT_DIR", "transport_results/gsm8k_tinyllava_llama")

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------- 1) 准备 GSM8K 文本（统一入口） --------------------
    texts = load_texts(
        data=args.data,  # 此处应为 "gsm8k"
        split=args.data_split,
        max_samples=args.max_samples,
    )

    # -------------------- 2) 抓 A 侧（TinyLLaVA / Llama3 / Qwen2-VL）的语言端模块激活 --------------------
    t_a = args.model_a_type.lower()
    if "tinyllava" in t_a:
        from activs_tinyllava_modules import get_tinyllava_block_activations_by_module
        print(f"[Info] A-side uses TinyLLaVA: {args.model_a_path}")
        x_per_layer, x_per_module = get_tinyllava_block_activations_by_module(
            model_path=args.model_a_path,
            texts=texts,
            conv_mode="qwen2_base",
            image_paths=None,  # 无图
            insert_image_token=None,  # 无图不插 <image>
            batch_size=args.batch_size_a,
            device=args.device_a,
        )
        a_label = f"TinyLLaVA({args.model_a_path})"
    elif t_a.startswith("llama3"):
        print(f"[Info] A-side uses Llama3: {args.model_a_path}")
        # 如果提供了 top_neuron_dir_a，使用带 neuron 的版本
        if args.top_neuron_dir_a:
            print(f"[Info] Using top-neuron mode with dir: {args.top_neuron_dir_a}")
            # 提前创建目录并验证
            try:
                os.makedirs(args.top_neuron_dir_a, exist_ok=True)
                if not os.path.exists(args.top_neuron_dir_a):
                    raise OSError(f"无法创建目录: {args.top_neuron_dir_a}")
                if not os.path.isdir(args.top_neuron_dir_a):
                    raise OSError(f"路径不是目录: {args.top_neuron_dir_a}")
                # 检查写入权限
                test_file = os.path.join(args.top_neuron_dir_a, ".write_test")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as e:
                    raise OSError(f"目录无写入权限: {args.top_neuron_dir_a}, 错误: {e}")
                print(f"[Info] Top neuron 目录已创建并验证: {args.top_neuron_dir_a}")
            except Exception as e:
                print(f"[ERROR] 无法创建或验证 top_neuron_dir: {e}", file=sys.stderr)
                traceback.print_exc()
                raise
            
            x_per_layer, x_per_module = get_llama3_block_activations_by_module_neuron(
                prompts=texts,
                model_name=args.model_a_path,
                batch_size=args.batch_size_a,
                device=args.device_a,
                top_neuron_dir=args.top_neuron_dir_a,
            )
        else:
            x_per_layer, x_per_module = get_llama3_block_activations_plain(
                prompts=texts,
                model_name=args.model_a_path,
                batch_size=args.batch_size_a,
                device=args.device_a,
            )
        a_label = f"Llama3({args.model_a_path})"
    elif "qwenvl" in t_a:
        from activs_qwenvl2_modules import get_qwenvl2_block_activations_by_module
        print(f"[Info] A-side uses Qwen2-VL: {args.model_a_path}")
        x_per_layer, x_per_module = get_qwenvl2_block_activations_by_module(
            model_name=args.model_a_path,
            texts=texts,
            image_paths=None,  # 可传等长图片路径列表启用多模态；此处默认纯文本
            batch_size=args.batch_size_a,
            device=args.device_a,
            # dtype=torch.bfloat16,  # 显存吃紧时可打开
            exclude_image_token=True,  # 默认屏蔽图像占位 token 的统计
        )
        a_label = f"Qwen2-VL({args.model_a_path})"
    else:
        raise ValueError(
            "data=gsm8k 时，model-a-type 仅支持包含 'tinyllava'、'llama3' 或 'qwenvl' 的字符串"
        )

    # -------------------- 3) 抓 B 侧（Qwen2 / Llama3 / Qwen2-VL）的模块激活（纯文本） --------------------
    t_b = args.model_b_type.lower()
    if t_b.startswith("llama3"):
        print(f"[Info] B-side uses Llama3: {args.model_b_path}")
        q_per_layer, q_per_module = get_llama3_block_activations_plain(
            prompts=texts,
            model_name=args.model_b_path,
            batch_size=args.batch_size_b,
            device=args.device_b,
        )
        b_label = f"Llama3({args.model_b_path})"

    elif t_b.startswith("qwen2") and "vl" not in t_b:
        # 纯文本 Qwen2
        print(f"[Info] B-side uses Qwen2: {args.model_b_path}")
        q_per_layer, q_per_module = get_qwen2_block_activations_by_module_neuron(
            prompts=texts,
            model_name=args.model_b_path,
            batch_size=args.batch_size_b,
            device=args.device_b,
        )
        b_label = f"Qwen2({args.model_b_path})"

    elif "qwenvl" in t_b:
        from activs_qwenvl2_modules import get_qwenvl2_block_activations_by_module
        # 新增：多模态 Qwen2-VL，当作纯文本用，屏蔽图像 token
        print(f"[Info] B-side uses Qwen2-VL: {args.model_b_path}")
        q_per_layer, q_per_module = get_qwenvl2_block_activations_by_module(
            model_name=args.model_b_path,
            texts=texts,
            image_paths=None,  # 此处仍然只用纯文本
            batch_size=args.batch_size_b,
            device=args.device_b,
            # dtype=torch.bfloat16,   # 显存吃紧时可打开
            exclude_image_token=True,  # 一致保持：不统计图像占位 token
        )
        b_label = f"Qwen2-VL({args.model_b_path})"

    else:
        raise ValueError(
            "data=gsm8k 时，model-b-type 仅支持包含 'llama3'、'qwen2' 或 'qwenvl' 的字符串"
        )

    # 覆盖度
    print_coverage(x_per_layer, a_label)
    print_coverage(q_per_layer, b_label)

    # -------------------- 4) 对所有模块批量计算 HOT，并保存（原代码A逻辑） --------------------
    BIG_FFN = {
        "fwd_up",
        "fwd_down",
        "up",
        "down",
        "fwd_up_pre",
        "fwd_down_pre",
        "up_pre",
        "down_pre",
    }

    BASE_MODULES = ["K", "Q", "V", "O"]  # 也可以加 "fwd_up","fwd_down"
    MODULES_WANTED = []
    for m in BASE_MODULES:
        MODULES_WANTED += [m, m + "_pre"]  # 同时跑 post / pre

    summary = {}

    for mod in MODULES_WANTED:
        device_local = "cuda:0"
        if mod in BIG_FFN:
            device_local = "cpu"  # 关键：把大矩阵的 HOT 放到 CPU

        try:
            X_list_raw = get_module_per_layer(x_per_layer, mod)  # A 侧
            Y_list_raw = get_module_per_layer(q_per_layer, mod)  # B 侧
        except KeyError:
            print(f"[Skip] {mod}: 缺少该键。")
            continue  # 某模块不存在，跳过

        # 诊断前 3 层
        print(f"\n[Diag] module={mod}")
        for li, (XA, YB) in enumerate(zip(X_list_raw[:3], Y_list_raw[:3])):
            if isinstance(XA, torch.Tensor) and isinstance(YB, torch.Tensor):
                print(
                    f"  layer {li}: A T×H={tuple(XA.shape)}, B T×H={tuple(YB.shape)}"
                )
            else:
                print(f"  layer {li}: A/B 其中至少一侧为空")

        # 过滤空层
        X_list, Y_list, kept = filter_nonempty_layer_pairs(X_list_raw, Y_list_raw)
        if len(kept) == 0:
            print(f"[WARN] {mod}: 所有层均为空，跳过 HOT。")
            continue
        if len(kept) < len(X_list_raw):
            dropped = [i for i in range(len(X_list_raw)) if i not in kept]
            print(f"[INFO] {mod}: 过滤掉空层 {dropped}，实际计算层数={len(kept)}")

        # 标准化（原代码A的 normalize_per_feature）
        X_list = [normalize_per_feature(x) for x in X_list]
        Y_list = [normalize_per_feature(y) for y in Y_list]

        # 跑 HOT（保持原代码A的 reg_inner / reg_outer）
        with torch.no_grad():
            Q_list_mod, C_layer_mod, _ = compute_Q_and_layer_costs(
                X_list, Y_list, reg_inner=0.03, device=device_local
            )
            P_mod = compute_P(C_layer_mod, reg_outer=0.1)

        # 保存（根据是否为 *_pre 决定文件名后缀）
        suffix = "" if not mod.endswith("_pre") else "_pre"
        mod_base = mod[:-4] if suffix == "_pre" else mod
        save_path = os.path.join(args.out_dir, f"hot_{mod_base}{suffix}.pt")
        save_dtype_str = getattr(args, "hot_save_dtype", "float32")
        save_dtype = torch.float16 if save_dtype_str == "float16" else torch.float32
        use_uint8 = save_dtype_str == "uint8"
        if use_uint8:
            q_saved, q_scales = [], []
            for row in Q_list_mod:
                q_row, s_row = [], []
                for q in row:
                    q = q.cpu().float()
                    scale = q.max().clamp_min(1e-8).item()
                    q_row.append((q / scale * 255.0).clamp(0.0, 255.0).round().to(torch.uint8))
                    s_row.append(scale)
                q_saved.append(q_row)
                q_scales.append(s_row)
        else:
            q_saved = [[q.cpu().to(save_dtype) for q in row] for row in Q_list_mod]
            q_scales = None
        out = {
            "module": mod,
            "P": P_mod.cpu().to(save_dtype),
            "Q_list": q_saved,
            "C_layer": C_layer_mod.cpu().to(save_dtype),
            **({"Q_scales": q_scales} if q_scales is not None else {}),
            "L": len(X_list),  # 参与计算的 A 侧层数（过滤后）
            "M": len(Y_list),  # 参与计算的 B 侧层数（过滤后）
            "kept_layer_idx": kept,  # 原层索引，便于对齐
            "notes": (
                "P 行和≈1/L；列和≈1/M；Q_list[ell][m] 形状 [n_Aℓ, n_Bm]；本文件为"
                + ("post（线性层之后）" if suffix == "" else "pre（线性层之前）")
                + f"；data=gsm8k; A={args.model_a_path}; B={args.model_b_path}"
            ),
        }
        
        # 确保输出目录存在
        try:
            os.makedirs(args.out_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] {mod}: 无法创建输出目录 {args.out_dir}: {e}", file=sys.stderr)
            continue
        
        # 保存文件，带异常处理
        try:
            torch.save(out, save_path)
            # 验证文件是否成功保存
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"文件保存后不存在: {save_path}")
            file_size = os.path.getsize(save_path)
            if file_size == 0:
                raise ValueError(f"保存的文件大小为0: {save_path}")
        except Exception as e:
            print(f"[ERROR] {mod}: 保存失败 -> {save_path}", file=sys.stderr)
            print(f"  错误详情: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

        # 控制台简报
        rs = P_mod.sum(dim=1).detach().cpu().tolist()
        cs = P_mod.sum(dim=0).detach().cpu().tolist()
        print(f"[{mod}] saved -> {save_path}")
        print(
            f"  L={len(X_list)}, M={len(Y_list)}, "
            f"P row sums≈ {rs[:3]}{'...' if len(rs) > 3 else ''}"
        )
        print(
            f"                               col sums≈ "
            f"{cs[:3]}{'...' if len(cs) > 3 else ''}"
        )

        summary[mod] = {"L": len(X_list), "M": len(Y_list), "kept": kept}

    print("Done. Summary:", summary)


# -------------------- main --------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("==================================================")
    print(f"[run_activs_and_hot] data = {args.data}")
    print(f"  model A path: {args.model_a_path}")
    print(f"  model B path: {args.model_b_path}")
    print(f"  out dir     : {args.out_dir}")
    print("==================================================")

    if args.data == "gsm8k":
        run_pipeline_gsm8k(args)
    else:  # malay / indonesian / eng / medical
        run_pipeline_llama3_text(args)


if __name__ == "__main__":
    main()
