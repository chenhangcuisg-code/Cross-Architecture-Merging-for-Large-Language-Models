# 评测结果提取脚本使用说明

## 脚本功能

`extract_eval_results.py` 是一个独立的评测结果提取脚本，用于解决原始训练脚本中评测结果提取不完整的问题。

### 主要功能
1. **扫描评测输出目录**：自动查找最新的评测结果文件
2. **解析评测结果**：从JSON结果文件中提取准确率等指标
3. **格式化输出**：按照lm-evaluation-harness的标准格式输出结果表格
4. **更新汇总日志**：允许覆盖已存在的旧记录，优先保留最新结果

## 解决的问题

### 原始问题
- 训练日志中显示的评测结果无法正确提取到汇总日志
- 去重逻辑过于严格，导致新结果无法覆盖旧记录
- 模型路径、时间戳等信息记录不正确
- **表格内容缺失**：ablation评测的结果表格是空的

### 解决方案
- ✅ **修复去重逻辑**：允许更新已存在的记录
- ✅ **正确解析JSON**：从结果文件中提取正确的配置信息
- ✅ **时间戳检查**：优先保留最新的评测结果
- ✅ **格式标准化**：确保输出格式与训练日志完全一致
- ✅ **支持双重评测**：正确处理原始评测（印尼语任务）和消融评测（通用benchmark）

## 使用方法

### 基本用法
```bash
cd /home/chenhang/optimal_trans
python3 extract_eval_results.py --task indonesian --run_label final_nolora_same_param_debugindo2
```

### 参数说明
- `--task`: 任务名称（默认：indonesian）
- `--run_label`: 运行标签（默认：final_nolora_same_param_debugindo2）
- `--lm_eval_repo`: lm-evaluation-harness仓库路径
- `--log_file`: 汇总日志文件路径
- `--force_update`: 强制更新已存在的记录

### 示例
```bash
# 提取indonesian任务的结果
python3 extract_eval_results.py --task indonesian --run_label final_nolora_same_param_debugindo2

# 强制更新所有记录
python3 extract_eval_results.py --force_update

# 指定自定义日志文件
python3 extract_eval_results.py --log_file /path/to/custom/log.txt
```

## 输出目录结构

脚本会扫描以下目录中的结果文件：
1. `output/` - ablation评测结果
2. `output_nolora_{run_label}/` - 原始评测结果

## 支持的评测任务

目前支持的印尼语评测任务：
- `arc_id`: AI2 Reasoning Challenge Indonesian
- `belebele_ind_Latn`: Belebele Indonesian
- `copal_id_colloquial`: COPAL Colloquial Indonesian
- `copal_id_standard`: COPAL Standard Indonesian
- `truthfulqa_id_mc1`: TruthfulQA Indonesian MC1
- `truthfulqa_id_mc2`: TruthfulQA Indonesian MC2
- `xcopa_id`: XCOPA Indonesian
- `xstorycloze_id`: XStoryCloze Indonesian

## 输出格式

脚本会生成与lm-evaluation-harness完全一致的表格格式：

```
2026-01-22:04:12:39 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
hf (pretrained=/path/to/model,dtype=float), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 8
|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_id           |      2|none  |     5|acc     |↑  |0.2513|±  |0.0127|
...
```

## 日志更新逻辑

1. **检查是否存在**：根据任务名、运行名、变体和评测类型查找已存在的记录
2. **比较时间戳**：如果存在新结果，会替换整个记录块
3. **保留最新**：始终保留最新的评测结果

## 注意事项

1. **备份重要数据**：运行前建议备份原有的汇总日志文件
2. **路径配置**：确保lm-evaluation-harness的输出路径配置正确
3. **权限检查**：确保对输出目录和日志文件有读写权限
4. **JSON格式**：脚本假设结果文件为标准JSON格式

## 故障排除

### 常见问题
1. **找不到结果文件**：检查输出目录路径是否正确
2. **解析错误**：确认JSON文件格式是否标准
3. **权限问题**：确保对相关目录有访问权限

### 日志位置
- 脚本运行日志会输出到控制台
- 更新结果会写入指定的汇总日志文件

## 技术细节

- **语言**: Python 3.8+
- **依赖**: 标准库（os, json, glob, argparse, datetime, pathlib）
- **兼容性**: 与现有的训练脚本完全兼容
